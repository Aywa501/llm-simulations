#!/usr/bin/env node
'use strict';

/**
 * download_router.js
 *
 * Purpose:
 *   Build a routing plan for heterogeneous data-download URLs in trials_pruned.json.
 *   The router decides, per URL, whether to use:
 *     - URL transformation / stable endpoint / HTML parsing (no Playwright)
 *     - Playwright with a reusable session
 *     - manual review
 *
 * Design notes derived from the user's scan outputs and HTML captures:
 *   - firstHostOnly=true, 32 unique hosts, click probing enabled in the scan metadata.
 *   - Major families in the scan summary: direct_file, openICPSR/ICPSR, Google Drive,
 *     Dataverse, World Bank, UK Data Service, plus singletons for AEA/OSF/Zenodo/
 *     Figshare/GitHub/Bitbucket/Dropbox/Dryad/Mendeley.
 *   - HTML captures expose concrete non-browser endpoints for several repositories:
 *       * Dryad: /dataset/downloadZip/... and /downloads/file_stream/<id>
 *       * Figshare: /ndownloader/files/<id>
 *       * Mendeley Data: /public-api/zip/<datasetId>/download/<version>
 *       * Dataverse pages expose persistentId and API links
 *       * AEA article pages expose a "Replication Package" link that may route to openICPSR
 *       * Google Drive file/sheet pages expose file IDs suitable for export endpoints
 *       * Old googledrive.com/host links can be dead (404)
 *
 * Outputs:
 *   <outdir>/download_router_plan.json
 *   <outdir>/download_router_plan.csv
 *   <outdir>/download_router_summary.json
 *   <outdir>/playwright_queue.json
 *   <outdir>/manual_queue.json
 *
 * This is intentionally a router/planner, not a full downloader. The next stage can
 * implement the family-specific downloaders against the plan this script produces.
 */

const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const FILE_EXT_RE = /\.(zip|7z|rar|tar|gz|bz2|xz|csv|tsv|txt|dta|sav|rda|rds|do|xlsx?|ods|json|xml|pdf)$/i;
const DOI_PREFIX_RE = /10\.\d{4,9}\/[A-Za-z0-9._;()\/:+-]+/;

function parseArgs(argv) {
  const args = {
    input: './trials_pruned.json',
    outdir: './router_out',
    htmlHints: null,
    allowAll: false,
    verifyDirect: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--input' && next) {
      args.input = next;
      i += 1;
    } else if (arg === '--outdir' && next) {
      args.outdir = next;
      i += 1;
    } else if (arg === '--html-hints' && next) {
      args.htmlHints = next;
      i += 1;
    } else if (arg === '--allow-all') {
      args.allowAll = true;
    } else if (arg === '--verify-direct') {
      args.verifyDirect = true;
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function printHelp() {
  console.log(`
Usage:
  node download_router.js --input ./trials_pruned.json --outdir ./router_out [--html-hints ./html_list.json]

Flags:
  --input         Path to trials_pruned.json
  --outdir        Output directory
  --html-hints    Optional path to html_list.json; used only to annotate evidence notes
  --allow-all     Include low-confidence families in the main plan instead of routing them to manual by default
  --verify-direct Mark direct-file links for HEAD/GET verification in the plan
`);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function safeUrl(raw) {
  try {
    return new URL(raw);
  } catch {
    return null;
  }
}

function normalizeHost(hostname) {
  if (!hostname) return '';
  return hostname.toLowerCase().replace(/^www\./, '');
}

function normalizeUrl(raw) {
  return String(raw || '').replace(/&amp;/g, '&').trim();
}

function extractDoiFromString(text) {
  const match = String(text || '').match(DOI_PREFIX_RE);
  return match ? match[0] : null;
}

function doiFamily(doi) {
  if (!doi) return null;
  const lower = doi.toLowerCase();
  if (lower.startsWith('10.7910/')) return 'dataverse';
  if (lower.startsWith('10.3886/')) return 'openicpsr_icpsr';
  if (lower.startsWith('10.17605/')) return 'osf';
  if (lower.startsWith('10.5281/')) return 'zenodo';
  if (lower.startsWith('10.6084/')) return 'figshare';
  if (lower.startsWith('10.5061/')) return 'dryad';
  if (lower.startsWith('10.17632/')) return 'mendeley_data';
  if (lower.startsWith('10.1257/')) return 'aea';
  if (lower.startsWith('10.5255/')) return 'uk_data_service';
  if (lower.startsWith('10.48529/')) return 'worldbank_microdata';
  if (lower.startsWith('10.1016/')) return 'publisher';
  if (lower.startsWith('10.1017/')) return 'publisher';
  return null;
}

function classifyUrl(rawUrl) {
  const url = safeUrl(normalizeUrl(rawUrl));
  if (!url) {
    return {
      family: 'invalid',
      host: '',
      doi: null,
      reason: 'Unparseable URL',
    };
  }

  const host = normalizeHost(url.hostname);
  const pathname = url.pathname || '';
  const doi = doiFamily(extractDoiFromString(decodeURIComponent(normalizeUrl(rawUrl)))) || doiFamily(extractDoiFromString(pathname));
  if (doi) {
    return { family: doi, host, doi: extractDoiFromString(normalizeUrl(rawUrl)) || extractDoiFromString(pathname), reason: 'DOI prefix classification' };
  }

  if (host.includes('dataverse') || host === 'dvn.iq.harvard.edu') return { family: 'dataverse', host, doi: null, reason: 'Dataverse host' };
  if (host === 'hdl.handle.net') return { family: 'hdl_handle', host, doi: null, reason: 'Handle resolver' };
  if (host.includes('openicpsr') || host.includes('icpsr.umich.edu') || host.startsWith('mcc.icpsr') || host === 'data.mcc.gov') return { family: 'openicpsr_icpsr', host, doi: null, reason: 'ICPSR/openICPSR host' };
  if (host.includes('aeaweb.org')) return { family: 'aea', host, doi: null, reason: 'AEA host' };
  if (host === 'osf.io') return { family: 'osf', host, doi: null, reason: 'OSF host' };
  if (host === 'zenodo.org') return { family: 'zenodo', host, doi: null, reason: 'Zenodo host' };
  if (host === 'github.com' || host === 'raw.githubusercontent.com') return { family: 'github', host, doi: null, reason: 'GitHub host' };
  if (host === 'bitbucket.org' || host === 'api.bitbucket.org') return { family: 'bitbucket', host, doi: null, reason: 'Bitbucket host' };
  if (host.endsWith('dropbox.com') || host === 'dropbox.com') return { family: 'dropbox', host, doi: null, reason: 'Dropbox host' };
  if (host === 'figshare.com') return { family: 'figshare', host, doi: null, reason: 'Figshare host' };
  if (host === 'datadryad.org') return { family: 'dryad', host, doi: null, reason: 'Dryad host' };
  if (host === 'data.mendeley.com') return { family: 'mendeley_data', host, doi: null, reason: 'Mendeley Data host' };
  if (host === 'drive.google.com' || host === 'docs.google.com' || host.endsWith('.googledrive.com') || host === 'googledrive.com') {
    return { family: 'google_drive', host, doi: null, reason: 'Google Drive/Docs host' };
  }
  if (host.includes('worldbank.org')) return { family: 'worldbank_microdata', host, doi: null, reason: 'World Bank host' };
  if (host.includes('ukdataservice.ac.uk') || host.includes('data-archive.ac.uk')) return { family: 'uk_data_service', host, doi: null, reason: 'UK Data Service host' };
  if (host === 'sites.google.com' || host === 'wixsite.com' || host.endsWith('.edu') || host.endsWith('.ac.uk') || host === 'ifpri.org' || host === 'erfdataportal.com') return { family: 'faculty_site', host, doi: null, reason: 'Institutional/faculty/ad hoc host' };
  if (host === 'mdrc.org' || host === 'younglives.org.uk' || host === 'understandingsociety.ac.uk' || host === 'ifo.de' || host === 'labnarasi.id') return { family: 'government_program', host, doi: null, reason: 'Government/program/info host' };
  if (host.includes('sciencedirect.com') || host.includes('journals.uchicago.edu') || host.includes('onlinelibrary.wiley.com') || host.includes('econometricsociety.org') || host.includes('cambridge.org') || host.includes('informs.org') || host.includes('oup.com') || host.includes('oxfordjournals.org') || host.includes('linkinghub.elsevier.com')) {
    return { family: 'publisher', host, doi: null, reason: 'Publisher/article host' };
  }
  if (FILE_EXT_RE.test(pathname)) return { family: 'direct_file', host, doi: null, reason: 'Direct file extension in URL path' };
  if (/\b(download|file|dataset|replication|data)\b/i.test(pathname)) return { family: 'direct_or_landing', host, doi: null, reason: 'Data-ish URL path' };
  return { family: 'other', host, doi: null, reason: 'Fallback classification' };
}

function buildDataversePlan(urlString) {
  const raw = normalizeUrl(urlString);
  const url = safeUrl(raw);
  const pid = url?.searchParams.get('persistentId') || extractDoiFromString(raw) || extractDoiFromString(decodeURIComponent(url?.pathname || ''));
  const origin = url?.origin || 'https://dataverse.harvard.edu';
  const candidates = [];
  if (pid) {
    candidates.push(`${origin}/api/datasets/:persistentId/?persistentId=${encodeURIComponent(pid)}`);
    candidates.push(`${origin}/api/access/dataset/:persistentId/?persistentId=${encodeURIComponent(pid)}`);
  }
  return {
    family: 'dataverse',
    method: 'http_api',
    usePlaywright: false,
    confidence: 'high',
    candidates,
    notes: [
      'Use persistentId-based Dataverse API routing; do not browser-click this family first.',
      'HTML evidence showed persistentId and Dataverse API/export links on page.',
    ],
    nextAction: 'Fetch metadata first, then resolve file IDs and download per file.',
  };
}

function buildHandlePlan(urlString) {
  const raw = normalizeUrl(urlString);
  const doi = extractDoiFromString(raw);
  if (doi) return buildRouterEntry(urlString, doiFamily(doi) || 'other');
  return {
    family: 'hdl_handle',
    method: 'http_resolve_then_reclassify',
    usePlaywright: false,
    confidence: 'medium',
    candidates: [raw],
    notes: ['Resolve the handle first, then reclassify the final URL.'],
    nextAction: 'HEAD/GET resolve final location, then re-route.',
  };
}

function buildDryadPlan(urlString) {
  const raw = normalizeUrl(urlString);
  const doi = extractDoiFromString(raw);
  const slug = doi ? doi.replace(/[/:.]/g, '_') : null;
  const candidates = [];
  if (slug) candidates.push(`https://datadryad.org/dataset/downloadZip/${slug}.zip`);
  return {
    family: 'dryad',
    method: 'html_parse_then_http',
    usePlaywright: false,
    confidence: 'high',
    candidates,
    notes: [
      'Dryad HTML capture exposed /dataset/downloadZip/... and /downloads/file_stream/<id> endpoints.',
      'Fetch the landing page once, parse the form action and individual file_stream links, then download via HTTP.',
    ],
    nextAction: 'Parse zip form action from HTML; fall back to individual file_stream links.',
  };
}

function buildFigsharePlan(urlString) {
  const raw = normalizeUrl(urlString);
  const m = raw.match(/figshare\.com\/articles\/(?:dataset|figure|media|files)\/[^/]+\/(\d+)/i) || raw.match(/10\.6084\/m9\.figshare\.(\d+)/i);
  const fileId = m ? m[1] : null;
  const candidates = [];
  if (fileId) candidates.push(`https://figshare.com/ndownloader/files/${fileId}`);
  return {
    family: 'figshare',
    method: 'html_parse_then_http',
    usePlaywright: false,
    confidence: 'high',
    candidates,
    notes: [
      'HTML capture exposed /ndownloader/files/<id> stable download links.',
      'Prefer parsing the landing page for ndownloader URLs instead of Playwright.',
    ],
    nextAction: 'GET landing page, extract /ndownloader/files/<id>, then download directly.',
  };
}

function buildMendeleyPlan(urlString) {
  const raw = normalizeUrl(urlString);
  const m = raw.match(/datasets\/([A-Za-z0-9]+)\/(\d+)/i) || raw.match(/10\.17632\/([A-Za-z0-9]+)\.(\d+)/i);
  const ds = m ? m[1] : null;
  const version = m ? m[2] : null;
  const candidates = [];
  if (ds && version) candidates.push(`https://data.mendeley.com/public-api/zip/${ds}/download/${version}`);
  return {
    family: 'mendeley_data',
    method: 'html_parse_then_http',
    usePlaywright: false,
    confidence: 'high',
    candidates,
    notes: [
      'HTML capture exposed /public-api/zip/<datasetId>/download/<version> and public-files dataset links.',
      'This family should be handled without Playwright unless access is unexpectedly restricted.',
    ],
    nextAction: 'Use the public zip endpoint first; fall back to individual public-files links.',
  };
}

function buildZenodoPlan(urlString) {
  const raw = normalizeUrl(urlString);
  const m = raw.match(/zenodo\.org\/(?:records|record)\/(\d+)/i) || raw.match(/10\.5281\/zenodo\.(\d+)/i);
  const recordId = m ? m[1] : null;
  return {
    family: 'zenodo',
    method: 'html_parse_then_http',
    usePlaywright: false,
    confidence: 'high',
    candidates: recordId ? [`https://zenodo.org/records/${recordId}`] : [raw],
    notes: [
      'Zenodo pages expose file URLs under /records/<id>/files/<name>?download=1.',
      'No Playwright is needed for public Zenodo records; parse and download over HTTP.',
    ],
    nextAction: 'Parse file anchors on the record page and download ?download=1 links.',
  };
}

function githubRawFromBlob(url) {
  const pathParts = url.pathname.split('/').filter(Boolean);
  if (url.hostname === 'raw.githubusercontent.com') return url.toString();
  const blobIdx = pathParts.indexOf('blob');
  if (pathParts.length >= 5 && blobIdx === 2) {
    const [owner, repo, _blob, ref, ...rest] = pathParts;
    return `https://raw.githubusercontent.com/${owner}/${repo}/${ref}/${rest.join('/')}`;
  }
  return null;
}

function buildGithubPlan(urlString) {
  const raw = normalizeUrl(urlString);
  const url = safeUrl(raw);
  const pathParts = url ? url.pathname.split('/').filter(Boolean) : [];
  const candidates = [];
  const rawCandidate = url ? githubRawFromBlob(url) : null;
  if (rawCandidate) candidates.push(rawCandidate);
  if (pathParts.length >= 2) {
    const [owner, repo] = pathParts;
    candidates.push(`https://github.com/${owner}/${repo}/archive/refs/heads/master.zip`);
    candidates.push(`https://github.com/${owner}/${repo}/archive/refs/heads/main.zip`);
  }
  return {
    family: 'github',
    method: rawCandidate ? 'url_transform' : 'html_parse_then_http',
    usePlaywright: false,
    confidence: 'high',
    candidates: dedupe(candidates),
    notes: [
      'GitHub scan exposed blob links for files and a public repository layout.',
      'Prefer raw/blob conversion for individual files; otherwise try repository archive URLs.',
    ],
    nextAction: rawCandidate ? 'Download via raw.githubusercontent.com.' : 'Parse page for blob/raw links or archive the repo.',
  };
}

function buildBitbucketPlan(urlString) {
  const raw = normalizeUrl(urlString);
  const url = safeUrl(raw);
  const pathParts = url ? url.pathname.split('/').filter(Boolean) : [];
  const candidates = [];
  if (pathParts.length >= 5 && pathParts[2] === 'src') {
    const [workspace, repo, _src, ref, ...rest] = pathParts;
    candidates.push(`https://bitbucket.org/${workspace}/${repo}/raw/${ref}/${rest.join('/')}`);
    candidates.push(`https://bitbucket.org/${workspace}/${repo}/downloads/`);
  }
  return {
    family: 'bitbucket',
    method: candidates.length ? 'url_transform' : 'html_parse_then_http',
    usePlaywright: false,
    confidence: 'medium',
    candidates: dedupe(candidates.length ? candidates : [raw]),
    notes: [
      'HTML capture exposed a repo file page and the /downloads/ route.',
      'Try raw-file transformation first; only escalate if the file is not directly reachable.',
    ],
    nextAction: 'Attempt raw URL first, then inspect /downloads/.',
  };
}

function buildDropboxPlan(urlString) {
  const raw = normalizeUrl(urlString);
  const url = safeUrl(raw);
  if (!url) return manualPlan('dropbox', raw, 'Invalid Dropbox URL');
  url.searchParams.set('dl', '1');
  return {
    family: 'dropbox',
    method: 'url_transform',
    usePlaywright: false,
    confidence: 'high',
    candidates: [url.toString()],
    notes: [
      'Dropbox public links can usually be downloaded by forcing dl=1.',
      'The scan exposed concrete file links inside the share, so HTML parse is a good fallback.',
    ],
    nextAction: 'Try dl=1 URL; if folder share, parse file anchors and transform each one.',
  };
}

function buildGoogleDrivePlan(urlString) {
  const raw = normalizeUrl(urlString);
  const url = safeUrl(raw);
  if (!url) return manualPlan('google_drive', raw, 'Invalid Google Drive URL');

  const host = normalizeHost(url.hostname);
  if (host.endsWith('.googledrive.com') || host === 'googledrive.com') {
    return {
      family: 'google_drive',
      method: 'manual_or_skip',
      usePlaywright: false,
      confidence: 'low',
      candidates: [raw],
      notes: [
        'Legacy googledrive.com/host links appeared as 404 in the HTML capture and scan.',
        'Do not spend Playwright time here until manually verified.',
      ],
      nextAction: 'Mark as legacy/dead unless a newer public share exists.',
    };
  }

  const fileMatch = raw.match(/drive\.google\.com\/file\/d\/([^/]+)/i);
  const sheetMatch = raw.match(/docs\.google\.com\/spreadsheets\/d\/([^/]+)/i);
  const docMatch = raw.match(/docs\.google\.com\/document\/d\/([^/]+)/i);
  const presentationMatch = raw.match(/docs\.google\.com\/presentation\/d\/([^/]+)/i);

  if (fileMatch) {
    return {
      family: 'google_drive',
      method: 'url_transform',
      usePlaywright: false,
      confidence: 'medium',
      candidates: [
        `https://drive.google.com/uc?export=download&id=${fileMatch[1]}`,
        `https://drive.google.com/file/d/${fileMatch[1]}/view`,
      ],
      notes: [
        'Google Drive public file pages expose a stable file ID.',
        'The scan observed one real browser download for this family; public-file transforms should be tried before Playwright.',
      ],
      nextAction: 'Attempt uc?export=download first. Escalate only if confirmation/token flow blocks it.',
    };
  }

  if (sheetMatch) {
    return {
      family: 'google_drive',
      method: 'url_transform',
      usePlaywright: false,
      confidence: 'medium',
      candidates: [
        `https://docs.google.com/spreadsheets/d/${sheetMatch[1]}/export?format=xlsx`,
        `https://docs.google.com/spreadsheets/d/${sheetMatch[1]}/export?format=csv`,
      ],
      notes: [
        'Google Sheets pages expose a stable document ID.',
        'Prefer export endpoints over Playwright for public spreadsheets.',
      ],
      nextAction: 'Attempt xlsx export, then csv if appropriate.',
    };
  }

  if (docMatch) {
    return {
      family: 'google_drive',
      method: 'url_transform',
      usePlaywright: false,
      confidence: 'medium',
      candidates: [
        `https://docs.google.com/document/d/${docMatch[1]}/export?format=docx`,
        `https://docs.google.com/document/d/${docMatch[1]}/export?format=pdf`,
      ],
      notes: ['Public Google Docs can usually be handled via export endpoints.'],
      nextAction: 'Try docx export first.',
    };
  }

  if (presentationMatch) {
    return {
      family: 'google_drive',
      method: 'url_transform',
      usePlaywright: false,
      confidence: 'medium',
      candidates: [`https://docs.google.com/presentation/d/${presentationMatch[1]}/export/pdf`],
      notes: ['Public Google Slides can usually be exported directly.'],
      nextAction: 'Try PDF export first.',
    };
  }

  return {
    family: 'google_drive',
    method: 'playwright',
    usePlaywright: true,
    confidence: 'medium',
    candidates: [raw],
    notes: [
      'Use Playwright only for folder-like, private, or confirmation-token-protected Drive flows.',
      'Do not send generic public file pages straight to the browser until transform attempts fail.',
    ],
    nextAction: 'Open with a persistent authenticated profile and click the visible download control.',
  };
}

function buildAeaPlan(urlString) {
  const raw = normalizeUrl(urlString);
  if (/_data\.zip(\?|$)/i.test(raw) || /\/data\/.+\.zip(\?|$)/i.test(raw)) {
    return {
      family: 'aea',
      method: 'direct_http',
      usePlaywright: false,
      confidence: 'high',
      candidates: [raw],
      notes: ['Direct AEA replication ZIPs should not go through Playwright.'],
      nextAction: 'Download directly via HTTP.',
    };
  }
  return {
    family: 'aea',
    method: 'html_parse_then_route',
    usePlaywright: false,
    confidence: 'high',
    candidates: [raw],
    notes: [
      'AEA article HTML exposed a Replication Package link and supplemental-material links.',
      'In the scan, the Replication Package route opened openICPSR in a popup.',
    ],
    nextAction: 'Fetch article HTML, extract the Replication Package href, then re-route that URL.',
  };
}

function buildOsfPlan(urlString) {
  const raw = normalizeUrl(urlString);
  return {
    family: 'osf',
    method: 'html_parse_then_http',
    usePlaywright: false,
    confidence: 'high',
    candidates: [raw],
    notes: [
      'The scan surfaced concrete OSF file links from the project page without requiring browser-only interaction.',
      'Use HTTP parse of project/file pages first; only escalate if the project is private or token-gated.',
    ],
    nextAction: 'Fetch project page, extract file links, then follow per-file download pages/endpoints.',
  };
}

function buildOpenIcpsrPlan(urlString) {
  const raw = normalizeUrl(urlString);
  return {
    family: 'openicpsr_icpsr',
    method: 'playwright',
    usePlaywright: true,
    confidence: 'high',
    candidates: [raw],
    notes: [
      'Scan summary showed this family is recurrent and login-likely for every sampled host.',
      'HTML captures exposed /download/terms flows and ICPSR pages with distribution contentURL metadata, but terms/profile/session handling is still the main risk.',
    ],
    nextAction: 'Use Playwright with a persistent authenticated profile. Accept terms once, then download project/archive or format-specific contentURL targets.',
  };
}

function buildUkDataServicePlan(urlString) {
  const raw = normalizeUrl(urlString);
  if (/reshare\.ukdataservice\.ac\.uk\/\d+\/\d+\/.+\.(zip|csv|txt|pdf)$/i.test(raw)) {
    return {
      family: 'uk_data_service',
      method: 'direct_http',
      usePlaywright: false,
      confidence: 'medium',
      candidates: [raw],
      notes: ['Direct Reshare file links can be downloaded without Playwright.'],
      nextAction: 'Download directly.',
    };
  }
  if (/reshare\.ukdataservice\.ac\.uk\/\d+\/?$/i.test(raw)) {
    return {
      family: 'uk_data_service',
      method: 'html_parse_then_http',
      usePlaywright: false,
      confidence: 'medium',
      candidates: [raw],
      notes: [
        'The scan found file-like zip links on at least one Reshare page.',
        'Try HTML parsing before escalating to Playwright.',
      ],
      nextAction: 'Parse the dataset page for direct file links (zip/documentation bundles).',
    };
  }
  return {
    family: 'uk_data_service',
    method: 'manual_or_browser',
    usePlaywright: false,
    confidence: 'low',
    candidates: [raw],
    notes: [
      'Generic UK Data Archive landing pages are not good download targets.',
      'Do not waste Playwright on root landing pages before finding the actual dataset/files page.',
    ],
    nextAction: 'Move to manual queue unless a concrete file/dataset page is available.',
  };
}

function buildWorldBankPlan(urlString) {
  const raw = normalizeUrl(urlString);
  return {
    family: 'worldbank_microdata',
    method: 'manual_or_playwright',
    usePlaywright: false,
    confidence: 'low',
    candidates: [raw],
    notes: [
      'HTML captures showed GET MICRODATA and distribution/contentUrl metadata, but access is often gated by registration or approval.',
      'Do not default this family into unattended Playwright runs.',
    ],
    nextAction: 'Keep in manual queue. Promote to supervised Playwright only after verifying the dataset is public-downloadable.',
  };
}

function buildPublisherPlan(urlString) {
  const raw = normalizeUrl(urlString);
  return {
    family: 'publisher',
    method: 'playwright',
    usePlaywright: true,
    confidence: 'medium',
    candidates: [raw],
    notes: [
      'Publisher platforms are session-heavy and supplementary materials are often nested behind article UIs.',
      'Use Playwright with institutional session reuse; do not try to normalize these with brittle URL transforms first.',
    ],
    nextAction: 'Open article page with persistent profile, find supplementary/replication links, then download.',
  };
}

function buildDirectFilePlan(urlString, verifyDirect = false) {
  const raw = normalizeUrl(urlString);
  return {
    family: 'direct_file',
    method: verifyDirect ? 'verify_then_http' : 'direct_http',
    usePlaywright: false,
    confidence: 'medium',
    candidates: [raw],
    notes: [
      'This bucket mixes true direct assets with dead faculty/institutional links.',
      'Do a HEAD/GET verification pass before spending browser time here.',
    ],
    nextAction: verifyDirect ? 'Issue HEAD, then fall back to GET if needed.' : 'Attempt direct download; mark dead if 404/timeout.',
  };
}

function buildOtherPlan(urlString) {
  return {
    family: 'other',
    method: 'manual_or_inspect',
    usePlaywright: false,
    confidence: 'low',
    candidates: [normalizeUrl(urlString)],
    notes: ['No stable routing rule matched.'],
    nextAction: 'Inspect manually or add a new family-specific handler.',
  };
}

function manualPlan(family, raw, note) {
  return {
    family,
    method: 'manual_or_skip',
    usePlaywright: false,
    confidence: 'low',
    candidates: [normalizeUrl(raw)],
    notes: [note],
    nextAction: 'Manual review.',
  };
}

function buildRouterEntry(urlString, family, verifyDirect = false, allowAll = false) {
  switch (family) {
    case 'dataverse': return buildDataversePlan(urlString);
    case 'hdl_handle': return buildHandlePlan(urlString);
    case 'dryad': return buildDryadPlan(urlString);
    case 'figshare': return buildFigsharePlan(urlString);
    case 'mendeley_data': return buildMendeleyPlan(urlString);
    case 'zenodo': return buildZenodoPlan(urlString);
    case 'github': return buildGithubPlan(urlString);
    case 'bitbucket': return buildBitbucketPlan(urlString);
    case 'dropbox': return buildDropboxPlan(urlString);
    case 'google_drive': return buildGoogleDrivePlan(urlString);
    case 'aea': return buildAeaPlan(urlString);
    case 'osf': return buildOsfPlan(urlString);
    case 'openicpsr_icpsr': return buildOpenIcpsrPlan(urlString);
    case 'uk_data_service': return buildUkDataServicePlan(urlString);
    case 'worldbank_microdata': return buildWorldBankPlan(urlString);
    case 'publisher': return buildPublisherPlan(urlString);
    case 'faculty_site': return manualPlan('faculty_site', urlString, 'Institutional/faculty/ad hoc host; verify manually unless it is a concrete direct file.');
    case 'government_program': return manualPlan('government_program', urlString, 'Government/program/info page; not a good unattended download target.');
    case 'direct_file':
    case 'direct_or_landing': return buildDirectFilePlan(urlString, verifyDirect);
    case 'other':
    default:
      return allowAll ? buildOtherPlan(urlString) : manualPlan(family, urlString, 'Low-confidence family; queued for manual review by default.');
  }
}

function dedupe(arr) {
  return [...new Set(arr.filter(Boolean))];
}

function htmlHintsSummary(htmlHints) {
  if (!Array.isArray(htmlHints)) return {};
  const summary = {
    count: htmlHints.length,
    hasDryadZipAction: 0,
    hasFigshareNdownloader: 0,
    hasMendeleyZipApi: 0,
    hasDataversePersistentId: 0,
    hasGoogleDriveFileId: 0,
    hasGoogleLegacy404: 0,
    hasAeaReplicationPackage: 0,
    hasBitbucketDownloadsRoute: 0,
  };
  for (const html of htmlHints) {
    const s = String(html || '');
    if (/\/dataset\/downloadZip\//i.test(s)) summary.hasDryadZipAction += 1;
    if (/figshare\.com\/ndownloader\/files\//i.test(s)) summary.hasFigshareNdownloader += 1;
    if (/\/public-api\/zip\//i.test(s)) summary.hasMendeleyZipApi += 1;
    if (/persistentId=doi:10\.7910/i.test(s)) summary.hasDataversePersistentId += 1;
    if (/drive\.google\.com\/file\/d\//i.test(s) || /docs\.google\.com\/spreadsheets\/d\//i.test(s)) summary.hasGoogleDriveFileId += 1;
    if (/Error 404 \(Not Found\)/i.test(s) && /googledrive/i.test(s)) summary.hasGoogleLegacy404 += 1;
    if (/Replication Package/i.test(s) && /aea/i.test(s)) summary.hasAeaReplicationPackage += 1;
    if (/bitbucket/i.test(s) && /\/downloads\//i.test(s)) summary.hasBitbucketDownloadsRoute += 1;
  }
  return summary;
}

function toCsv(rows) {
  const headers = [
    'rowId','rctId','title','sourceUrl','family','method','usePlaywright','confidence','host','classificationReason','candidateCount','candidates','nextAction','notes'
  ];
  const escape = (value) => {
    const text = String(value ?? '');
    if (/[",\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
    return text;
  };
  const lines = [headers.join(',')];
  for (const row of rows) {
    lines.push(headers.map((h) => escape(row[h])).join(','));
  }
  return `${lines.join('\n')}\n`;
}

function summarize(planRows, htmlHintStats, args) {
  const byFamily = {};
  const byMethod = {};
  let playwrightCount = 0;
  let manualCount = 0;

  for (const row of planRows) {
    byFamily[row.family] = (byFamily[row.family] || 0) + 1;
    byMethod[row.method] = (byMethod[row.method] || 0) + 1;
    if (row.usePlaywright) playwrightCount += 1;
    if (/manual/i.test(row.method)) manualCount += 1;
  }

  return {
    createdAt: new Date().toISOString(),
    input: args.input,
    outdir: args.outdir,
    totalRows: planRows.length,
    playwrightCount,
    manualCount,
    byFamily,
    byMethod,
    htmlHintStats,
    strategyPolicy: {
      httpFamilies: ['dataverse','dryad','figshare','mendeley_data','zenodo','github','bitbucket','dropbox','osf','aea_direct','google_drive_public','direct_file_verified','uk_data_service_reshare'],
      playwrightFamilies: ['openicpsr_icpsr','publisher','google_drive_private_or_confirmation'],
      manualFamilies: ['worldbank_microdata','generic_uk_data_service_landing','legacy_googledrive_host','unknown_other'],
    },
  };
}

function main() {
  const args = parseArgs(process.argv);
  ensureDir(args.outdir);

  const trials = readJson(args.input);
  const htmlHints = args.htmlHints ? readJson(args.htmlHints) : null;
  const htmlHintStats = htmlHintsSummary(htmlHints);

  const rows = [];
  for (const [rowId, value] of Object.entries(trials)) {
    const sourceUrl = normalizeUrl(value['Public Data URL'] || value.public_data_url || value.url || '');
    if (!sourceUrl) continue;
    const classification = classifyUrl(sourceUrl);
    const plan = buildRouterEntry(sourceUrl, classification.family, args.verifyDirect, args.allowAll);
    rows.push({
      rowId,
      rctId: value['RCT ID'] || value.rct_id || '',
      title: value['Title'] || value.title || '',
      sourceUrl,
      host: classification.host,
      family: plan.family,
      method: plan.method,
      usePlaywright: plan.usePlaywright,
      confidence: plan.confidence,
      classificationReason: classification.reason,
      candidateCount: plan.candidates.length,
      candidates: plan.candidates.join(' | '),
      nextAction: plan.nextAction,
      notes: plan.notes.join(' | '),
      strategy: plan,
    });
  }

  rows.sort((a, b) => {
    if (a.usePlaywright !== b.usePlaywright) return Number(a.usePlaywright) - Number(b.usePlaywright);
    return a.family.localeCompare(b.family);
  });

  const summary = summarize(rows, htmlHintStats, args);
  const playwrightQueue = rows.filter((r) => r.usePlaywright).map(stripStrategy);
  const manualQueue = rows.filter((r) => /manual/i.test(r.method)).map(stripStrategy);
  const planJson = rows.map(stripStrategy);

  fs.writeFileSync(path.join(args.outdir, 'download_router_plan.json'), JSON.stringify(planJson, null, 2));
  fs.writeFileSync(path.join(args.outdir, 'download_router_plan.csv'), toCsv(rows));
  fs.writeFileSync(path.join(args.outdir, 'download_router_summary.json'), JSON.stringify(summary, null, 2));
  fs.writeFileSync(path.join(args.outdir, 'playwright_queue.json'), JSON.stringify(playwrightQueue, null, 2));
  fs.writeFileSync(path.join(args.outdir, 'manual_queue.json'), JSON.stringify(manualQueue, null, 2));

  console.log(`Wrote ${rows.length} routed rows to ${args.outdir}`);
  console.log(`Playwright queue: ${playwrightQueue.length}`);
  console.log(`Manual queue: ${manualQueue.length}`);
}

function stripStrategy(row) {
  const copy = { ...row };
  delete copy.strategy;
  return copy;
}

main();
