#!/usr/bin/env node
/**
 * Scan AEA trial data URLs for likely automated download pathways.
 *
 * Defaults:
 *   - Visits only the first eligible instance of each host
 *   - Saves BOTH HTML snapshots and PNG screenshots for every scanned page
 *
 * Typical usage:
 *   node scan_download_pathways.js \
 *     --input ./trials_pruned.json \
 *     --outdir ./scan_out \
 *     --profile ./profiles/berkeley \
 *     --headed \
 *     --probe-clicks
 */

const fs = require('fs/promises');
const path = require('path');
let chromium;

function printHelpAndExit() {
  console.log(`
scan_download_pathways.js

Options:
  --input PATH                  Input JSON file. Default: trials_pruned.json
  --outdir PATH                 Output directory. Default: scan_out
  --profile PATH                Persistent browser profile dir. Default: playwright_profile
  --headed                      Run with visible browser
  --probe-clicks                Click-probe the top candidate controls
  --no-moderate                 Exclude moderate-complexity host families
  --allow-all                   Disable host-family filtering
  --all-instances               Visit every eligible URL instead of first host only
  --timeout-ms N                Navigation timeout. Default: 30000
  --action-timeout-ms N         Click/probe timeout. Default: 8000
  --max-candidates N            Max candidate controls saved per page. Default: 8
  --max-probe-candidates N      Max controls to click-probe. Default: 3
  --limit N                     Scan only the first N eligible records
  --no-save-html                Disable HTML snapshot capture
  --no-save-screenshots         Disable PNG screenshot capture
`);
  process.exit(0);
}

function parseArgs(argv) {
  const out = {
    input: 'trials_pruned.json',
    outdir: 'scan_out',
    profile: 'playwright_profile',
    headed: false,
    probeClicks: false,
    includeModerate: true,
    allowAll: false,
    firstHostOnly: true,
    timeoutMs: 30000,
    actionTimeoutMs: 8000,
    maxCandidates: 8,
    maxProbeCandidates: 3,
    limit: null,
    saveHtml: true,
    saveScreenshots: true,
  };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--input') out.input = next, i++;
    else if (arg === '--outdir') out.outdir = next, i++;
    else if (arg === '--profile') out.profile = next, i++;
    else if (arg === '--timeout-ms') out.timeoutMs = Number(next), i++;
    else if (arg === '--action-timeout-ms') out.actionTimeoutMs = Number(next), i++;
    else if (arg === '--max-candidates') out.maxCandidates = Number(next), i++;
    else if (arg === '--max-probe-candidates') out.maxProbeCandidates = Number(next), i++;
    else if (arg === '--limit') out.limit = Number(next), i++;
    else if (arg === '--headed') out.headed = true;
    else if (arg === '--probe-clicks') out.probeClicks = true;
    else if (arg === '--no-moderate') out.includeModerate = false;
    else if (arg === '--allow-all') out.allowAll = true;
    else if (arg === '--all-instances') out.firstHostOnly = false;
    else if (arg === '--no-save-html') out.saveHtml = false;
    else if (arg === '--no-save-screenshots') out.saveScreenshots = false;
    else if (arg === '--help' || arg === '-h') printHelpAndExit();
    else console.warn(`Ignoring unknown argument: ${arg}`);
  }
  return out;
}

function splitUrls(raw) {
  if (!raw || typeof raw !== 'string') return [];
  return raw
    .replace(/\r/g, '\n')
    .split(/[\n\s,;]+(?=https?:\/\/)/g)
    .map(s => s.trim())
    .filter(Boolean);
}

function normalizeHost(urlString) {
  try {
    return new URL(urlString).hostname.toLowerCase();
  } catch {
    return '';
  }
}

function extFromUrl(urlString) {
  try {
    const pathname = new URL(urlString).pathname.toLowerCase();
    const m = pathname.match(/\.([a-z0-9]{1,8})$/);
    return m ? m[1] : null;
  } catch {
    return null;
  }
}

const DIRECT_FILE_EXTS = new Set([
  'zip', '7z', 'rar', 'tar', 'gz', 'tgz',
  'csv', 'tsv', 'txt', 'tab', 'dat',
  'dta', 'rds', 'rdata', 'sav', 'por',
  'xlsx', 'xls', 'ods',
  'json', 'jsonl', 'parquet',
  'do', 'r', 'py', 'ipynb', 'm', 'pdf'
]);

function classifyUrl(urlString) {
  const host = normalizeHost(urlString);
  const ext = extFromUrl(urlString);

  const matchers = [
    { family: 'dataverse', tier: 'easy', feasible: true, re: /(^|\.)dataverse\./ },
    { family: 'doi_resolver', tier: 'easy', feasible: true, re: /(^|\.)doi\.org$|(^|\.)dx\.doi\.org$/ },
    { family: 'aea', tier: 'easy', feasible: true, re: /(^|\.)aeaweb\.org$|(^|\.)www-aeaweb-org\./ },
    { family: 'osf', tier: 'easy', feasible: true, re: /(^|\.)osf\.io$/ },
    { family: 'github', tier: 'easy', feasible: true, re: /(^|\.)github\.com$/ },
    { family: 'bitbucket', tier: 'easy', feasible: true, re: /(^|\.)bitbucket\.org$/ },
    { family: 'dropbox', tier: 'easy', feasible: true, re: /(^|\.)dropbox\.com$/ },
    { family: 'zenodo', tier: 'easy', feasible: true, re: /(^|\.)zenodo\.org$/ },
    { family: 'figshare', tier: 'easy', feasible: true, re: /(^|\.)figshare\.com$/ },
    { family: 'dryad', tier: 'easy', feasible: true, re: /(^|\.)datadryad\.org$/ },
    { family: 'mendeley_data', tier: 'easy', feasible: true, re: /(^|\.)data\.mendeley\.com$/ },

    { family: 'openicpsr_icpsr', tier: 'moderate', feasible: true, re: /(^|\.)openicpsr\.org$|(^|\.)icpsr\.umich\.edu$/ },
    { family: 'worldbank_microdata', tier: 'moderate', feasible: true, re: /(^|\.)microdata\.worldbank\.org$|(^|\.)datacatalog\.worldbank\.org$|(^|\.)data\.mcc\.gov$/ },
    { family: 'google_drive', tier: 'moderate', feasible: true, re: /(^|\.)drive\.google\.com$|(^|\.)docs\.google\.com$|(^|\.)googledrive\.com$/ },
    { family: 'uk_data_service', tier: 'moderate', feasible: true, re: /(^|\.)ukdataservice\.ac\.uk$|(^|\.)data-archive\.ac\.uk$/ },
    { family: 'handle_resolver', tier: 'moderate', feasible: true, re: /(^|\.)handle\.net$/ },
  ];

  for (const m of matchers) {
    if (m.re.test(host)) return { ...m, host, ext };
  }
  if (ext && DIRECT_FILE_EXTS.has(ext)) {
    return { family: 'direct_file', tier: 'easy', feasible: true, host, ext };
  }
  return { family: 'other', tier: 'hard', feasible: false, host, ext };
}

function makeRows(rawJson, opts) {
  const rows = [];
  const seenHosts = new Set();

  for (const [rowId, item] of Object.entries(rawJson)) {
    const urls = splitUrls(item['Public Data URL']);
    for (const sourceUrl of urls) {
      const c = classifyUrl(sourceUrl);
      const eligible = opts.allowAll || (c.feasible && (opts.includeModerate || c.tier === 'easy'));
      if (!eligible) continue;

      const hostKey = c.host || '__missing_host__';
      if (opts.firstHostOnly && seenHosts.has(hostKey)) continue;
      seenHosts.add(hostKey);

      rows.push({
        rowId,
        rctId: item['RCT ID'] || '',
        title: item['Title'] || '',
        sourceUrl,
        rawHost: c.host,
        initialFamily: c.family,
        initialTier: c.tier,
      });
    }
  }

  return opts.limit ? rows.slice(0, opts.limit) : rows;
}

function slugify(input, maxLen = 90) {
  return String(input || 'untitled')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
    .slice(0, maxLen) || 'untitled';
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function safeGoto(page, url, timeoutMs) {
  const result = { ok: false, status: null, finalUrl: null, error: null };
  try {
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    try {
      await page.waitForLoadState('networkidle', { timeout: 3000 });
    } catch {}
    result.ok = true;
    result.status = response ? response.status() : null;
    result.finalUrl = page.url();
  } catch (err) {
    result.error = err.message;
    result.finalUrl = page.url() || url;
  }
  return result;
}

function scoreCandidate(candidate) {
  let score = 0;
  const text = `${candidate.text || ''} ${candidate.href || ''} ${candidate.action || ''}`.toLowerCase();

  const goodTerms = [
    [/download/, 6],
    [/replication/, 5],
    [/supplement/, 4],
    [/data/, 4],
    [/dataset/, 5],
    [/files?/, 3],
    [/archive/, 3],
    [/code/, 2],
    [/zip|csv|stata|dta|rdata|rds|xlsx|xls|sav|pdf/, 5],
    [/raw/, 2],
    [/export/, 3],
    [/get microdata/, 6],
    [/access data/, 5],
    [/download this project/, 7],
  ];
  for (const [re, weight] of goodTerms) if (re.test(text)) score += weight;

  const badTerms = [
    [/login|log in|sign in|register|request access|apply|subscribe/, -6],
    [/logout|sign out|delete|remove|cookie|privacy|terms/, -8],
    [/citation|cite|share|tweet|facebook|linkedin/, -4],
    [/abstract|references|appendix/, -3],
  ];
  for (const [re, weight] of badTerms) if (re.test(text)) score += weight;

  if (candidate.href && /\.(zip|csv|tsv|xlsx?|ods|dta|sav|rds|rdata|jsonl?|parquet|do|r|py|ipynb|pdf)(\?|#|$)/i.test(candidate.href)) score += 10;
  if (candidate.downloadAttr) score += 6;
  if (candidate.targetBlank) score += 1;
  if (candidate.tagName === 'a' && candidate.href) score += 2;
  if (candidate.tagName === 'form') score += 2;
  if (candidate.visible) score += 2;

  return score;
}

async function collectPageSignals(page, maxCandidates) {
  return page.evaluate(({ maxCandidates }) => {
    const goodText = /(download|replication|data|dataset|supplement|files?|archive|raw|get microdata|access data|code|stata|csv|zip|xlsx|rdata|rds|sav|pdf)/i;
    const badText = /(sign in|login|log in|logout|sign out|register|privacy|cookie|terms|tweet|share)/i;

    const isVisible = (el) => {
      const style = window.getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      return style &&
        style.visibility !== 'hidden' &&
        style.display !== 'none' &&
        rect.width > 0 &&
        rect.height > 0;
    };

    const cssPath = (el) => {
      if (!(el instanceof Element)) return null;
      if (el.id) return `#${CSS.escape(el.id)}`;
      const parts = [];
      let node = el;
      while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 6) {
        let selector = node.nodeName.toLowerCase();
        if (node.classList && node.classList.length) {
          const stable = Array.from(node.classList)
            .filter(c => /^[a-zA-Z0-9_-]+$/.test(c))
            .slice(0, 2);
          if (stable.length) selector += '.' + stable.join('.');
        }
        const parent = node.parentElement;
        if (parent) {
          const siblings = Array.from(parent.children).filter(x => x.nodeName === node.nodeName);
          if (siblings.length > 1) selector += `:nth-of-type(${siblings.indexOf(node) + 1})`;
        }
        parts.unshift(selector);
        node = parent;
      }
      return parts.join(' > ');
    };

    const candidates = [];
    const nodes = document.querySelectorAll('a[href], button, [role="button"], input[type="button"], input[type="submit"], form');
    let idx = 0;
    for (const node of nodes) {
      const text = (
        node.innerText ||
        node.textContent ||
        node.getAttribute('aria-label') ||
        node.getAttribute('title') ||
        node.getAttribute('value') ||
        ''
      ).replace(/\s+/g, ' ').trim();

      const href = node.tagName.toLowerCase() === 'a' ? node.href : null;
      const action = node.tagName.toLowerCase() === 'form' ? (node.getAttribute('action') || null) : null;
      const targetBlank = (node.getAttribute('target') || '').toLowerCase() === '_blank';
      const downloadAttr = node.hasAttribute('download');
      const visible = isVisible(node);
      const tagName = node.tagName.toLowerCase();

      const combined = `${text} ${href || ''} ${action || ''}`;
      if (!visible) continue;
      if (!(goodText.test(combined) || /\.(zip|csv|tsv|xlsx?|ods|dta|sav|rds|rdata|jsonl?|parquet|pdf)(\?|#|$)/i.test(href || ''))) continue;
      if (badText.test(combined) && !goodText.test(combined)) continue;

      candidates.push({
        scanIndex: idx++,
        selector: cssPath(node),
        tagName,
        text: text.slice(0, 300),
        href,
        action,
        visible,
        downloadAttr,
        targetBlank,
      });
    }

    const bodyText = (document.body?.innerText || '').replace(/\s+/g, ' ').slice(0, 5000);
    const loginLikely = /(sign in|login|log in|institutional access|access through your institution|shibboleth|single sign-on|duo|request access|apply for access)/i.test(bodyText);
    const jsRequired = /(enable javascript|javascript is required)/i.test(bodyText);

    return {
      title: document.title,
      pageTextSnippet: bodyText.slice(0, 1200),
      loginLikely,
      jsRequired,
      fileLikeLinks: Array.from(document.querySelectorAll('a[href]'))
        .map(a => a.href)
        .filter(href => /\.(zip|csv|tsv|xlsx?|ods|dta|sav|rds|rdata|jsonl?|parquet|do|r|py|ipynb|pdf)(\?|#|$)/i.test(href))
        .slice(0, 25),
      forms: Array.from(document.forms).slice(0, 10).map((form, i) => ({
        formIndex: i,
        action: form.action || null,
        method: form.method || 'get',
        hasPassword: !!form.querySelector('input[type="password"]'),
      })),
      candidates: candidates.slice(0, maxCandidates),
    };
  }, { maxCandidates });
}

async function probeCandidate(context, baseUrl, candidate, opts) {
  const probe = {
    selector: candidate.selector,
    text: candidate.text,
    href: candidate.href,
    outcome: 'none',
    urlAfter: null,
    popupUrl: null,
    download: null,
    error: null,
  };

  const page = await context.newPage();
  page.setDefaultTimeout(opts.actionTimeoutMs);

  try {
    await safeGoto(page, baseUrl, opts.timeoutMs);

    let locator = null;
    if (candidate.selector) {
      locator = page.locator(candidate.selector).first();
      const count = await locator.count().catch(() => 0);
      if (!count) locator = null;
    }
    if (!locator && candidate.href) {
      const escapedHref = candidate.href.replace(/"/g, '\\"');
      locator = page.locator(`a[href="${escapedHref}"]`).first();
      const count = await locator.count().catch(() => 0);
      if (!count) locator = null;
    }
    if (!locator && candidate.text) {
      locator = page.getByText(candidate.text, { exact: false }).first();
      const count = await locator.count().catch(() => 0);
      if (!count) locator = null;
    }
    if (!locator) {
      probe.outcome = 'locator_not_found';
      await page.close().catch(() => {});
      return probe;
    }

    const downloadPromise = page.waitForEvent('download', { timeout: opts.actionTimeoutMs }).catch(() => null);
    const popupPromise = page.waitForEvent('popup', { timeout: opts.actionTimeoutMs }).catch(() => null);

    const beforeUrl = page.url();
    await locator.scrollIntoViewIfNeeded().catch(() => {});
    await locator.click({ timeout: opts.actionTimeoutMs }).catch(async () => {
      await locator.click({ timeout: opts.actionTimeoutMs, force: true });
    });

    const [download, popup] = await Promise.all([downloadPromise, popupPromise]);

    if (download) {
      probe.outcome = 'download';
      probe.download = {
        url: download.url(),
        suggestedFilename: download.suggestedFilename(),
        failure: await download.failure().catch(() => null),
      };
      await download.delete().catch(async () => {
        await download.cancel().catch(() => {});
      });
    }
    if (popup) {
      await popup.waitForLoadState('domcontentloaded', { timeout: opts.actionTimeoutMs }).catch(() => {});
      probe.popupUrl = popup.url();
      await popup.close().catch(() => {});
      if (probe.outcome === 'none') probe.outcome = 'popup';
    }

    await page.waitForTimeout(1200);
    probe.urlAfter = page.url();
    if (probe.outcome === 'none' && probe.urlAfter !== beforeUrl) probe.outcome = 'navigation';
    if (probe.outcome === 'none') probe.outcome = 'click_no_observable_effect';
  } catch (err) {
    probe.error = err.message;
    probe.outcome = 'error';
  } finally {
    await page.close().catch(() => {});
  }

  return probe;
}

async function scanOne(context, row, opts, artifactDirs) {
  const page = await context.newPage();
  page.setDefaultTimeout(opts.timeoutMs);

  const networkHints = [];
  const directDownloads = [];
  const popupUrls = [];
  const consoleErrors = [];
  const pageErrors = [];
  const dialogs = [];

  page.on('dialog', d => {
    dialogs.push({ type: d.type(), message: d.message() });
    d.dismiss().catch(() => {});
  });
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text().slice(0, 500));
  });
  page.on('pageerror', err => pageErrors.push(String(err.message || err).slice(0, 500)));
  page.on('popup', popup => {
    popupUrls.push(popup.url());
    popup.close().catch(() => {});
  });
  page.on('download', async download => {
    directDownloads.push({
      url: download.url(),
      suggestedFilename: download.suggestedFilename(),
      failure: await download.failure().catch(() => null),
    });
    await download.delete().catch(async () => {
      await download.cancel().catch(() => {});
    });
  });
  page.on('response', async response => {
    try {
      const url = response.url();
      const headers = await response.allHeaders();
      const cd = headers['content-disposition'] || '';
      const ct = headers['content-type'] || '';
      if (
        /attachment/i.test(cd) ||
        /\.(zip|csv|tsv|xlsx?|ods|dta|sav|rds|rdata|jsonl?|parquet|do|r|py|ipynb|pdf)(\?|#|$)/i.test(url) ||
        /(application\/zip|application\/x-zip|application\/octet-stream|text\/csv|application\/pdf)/i.test(ct)
      ) {
        networkHints.push({
          url,
          status: response.status(),
          contentType: ct,
          contentDisposition: cd,
        });
      }
    } catch {}
  });

  const nav = await safeGoto(page, row.sourceUrl, opts.timeoutMs);
  let signals = null;
  let probed = [];
  let finalFamily = row.initialFamily;
  let finalTier = row.initialTier;

  if (nav.ok) {
    try {
      signals = await collectPageSignals(page, opts.maxCandidates);
      if (signals?.candidates?.length) {
        signals.candidates = signals.candidates
          .map(c => ({ ...c, score: scoreCandidate(c) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, opts.maxCandidates);
      }
    } catch (err) {
      signals = { error: `collectPageSignals failed: ${err.message}` };
    }

    const resolved = classifyUrl(page.url());
    finalFamily = resolved.family;
    finalTier = resolved.tier;

    const baseName = `${slugify(row.rawHost || 'host')}__${slugify(row.rctId || row.rowId)}`;

    if (opts.saveHtml) {
      const htmlPath = path.join(artifactDirs.html, `${baseName}.html`);
      await fs.writeFile(htmlPath, await page.content(), 'utf8').catch(() => {});
    }

    if (opts.saveScreenshots) {
      const pngPath = path.join(artifactDirs.screenshots, `${baseName}.png`);
      await page.screenshot({ path: pngPath, fullPage: true }).catch(() => {});
    }

    if (opts.probeClicks && signals?.candidates?.length) {
      for (const candidate of signals.candidates.slice(0, opts.maxProbeCandidates)) {
        probed.push(await probeCandidate(context, page.url(), candidate, opts));
      }
    }
  }

  await page.close().catch(() => {});

  return {
    rowId: row.rowId,
    rctId: row.rctId,
    title: row.title,
    sourceUrl: row.sourceUrl,
    rawHost: row.rawHost,
    initialFamily: row.initialFamily,
    initialTier: row.initialTier,
    finalUrl: nav.finalUrl,
    finalHost: normalizeHost(nav.finalUrl || row.sourceUrl),
    finalFamily,
    finalTier,
    status: nav.status,
    navigationOk: nav.ok,
    navigationError: nav.error,
    pageTitle: signals?.title || null,
    loginLikely: signals?.loginLikely || false,
    jsRequired: signals?.jsRequired || false,
    pageTextSnippet: signals?.pageTextSnippet || null,
    fileLikeLinks: signals?.fileLikeLinks || [],
    forms: signals?.forms || [],
    candidates: signals?.candidates || [],
    networkHints: networkHints.slice(0, 25),
    directDownloads,
    popupUrls,
    dialogs,
    consoleErrors: consoleErrors.slice(0, 10),
    pageErrors: pageErrors.slice(0, 10),
    probed,
  };
}

function summarize(results) {
  const byFamily = {};
  for (const item of results) {
    const k = item.finalFamily || item.initialFamily || 'unknown';
    if (!byFamily[k]) {
      byFamily[k] = {
        count: 0,
        navOk: 0,
        loginLikely: 0,
        jsRequired: 0,
        hasFileLikeLinks: 0,
        hasNetworkHints: 0,
        hasCandidates: 0,
        probeDownload: 0,
        probeNavigation: 0,
        probePopup: 0,
      };
    }
    const b = byFamily[k];
    b.count++;
    if (item.navigationOk) b.navOk++;
    if (item.loginLikely) b.loginLikely++;
    if (item.jsRequired) b.jsRequired++;
    if ((item.fileLikeLinks || []).length) b.hasFileLikeLinks++;
    if ((item.networkHints || []).length || (item.directDownloads || []).length) b.hasNetworkHints++;
    if ((item.candidates || []).length) b.hasCandidates++;
    for (const p of item.probed || []) {
      if (p.outcome === 'download') b.probeDownload++;
      if (p.outcome === 'navigation') b.probeNavigation++;
      if (p.outcome === 'popup') b.probePopup++;
    }
  }
  return Object.entries(byFamily)
    .sort((a, b) => b[1].count - a[1].count)
    .map(([family, stats]) => ({ family, ...stats }));
}

async function writeOutputs(outdir, results) {
  const summary = summarize(results);
  await fs.writeFile(path.join(outdir, 'scan_results.json'), JSON.stringify(results, null, 2), 'utf8');
  await fs.writeFile(path.join(outdir, 'scan_results.jsonl'), results.map(r => JSON.stringify(r)).join('\n'), 'utf8');

  const csvRows = [[
    'rct_id', 'title', 'source_url', 'final_url', 'initial_family', 'final_family',
    'navigation_ok', 'status', 'login_likely', 'js_required',
    'candidate_count', 'filelike_link_count', 'network_hint_count',
    'probe_outcomes'
  ].join(',')];

  for (const r of results) {
    const probeOutcomes = (r.probed || []).map(p => p.outcome).join('|');
    const fields = [
      r.rctId,
      r.title,
      r.sourceUrl,
      r.finalUrl || '',
      r.initialFamily,
      r.finalFamily,
      String(r.navigationOk),
      r.status ?? '',
      String(r.loginLikely),
      String(r.jsRequired),
      String((r.candidates || []).length),
      String((r.fileLikeLinks || []).length),
      String((r.networkHints || []).length + (r.directDownloads || []).length),
      probeOutcomes,
    ].map(v => `"${String(v).replace(/"/g, '""')}"`);
    csvRows.push(fields.join(','));
  }

  await fs.writeFile(path.join(outdir, 'scan_results.csv'), csvRows.join('\n'), 'utf8');
  await fs.writeFile(path.join(outdir, 'summary_by_family.json'), JSON.stringify(summary, null, 2), 'utf8');
}

async function main() {
  const opts = parseArgs(process.argv);
  ({ chromium } = require('playwright'));

  await ensureDir(opts.outdir);
  await ensureDir(opts.profile);

  const artifactDirs = {
    html: path.join(opts.outdir, 'html'),
    screenshots: path.join(opts.outdir, 'screenshots'),
  };
  if (opts.saveHtml) await ensureDir(artifactDirs.html);
  if (opts.saveScreenshots) await ensureDir(artifactDirs.screenshots);

  const raw = JSON.parse(await fs.readFile(opts.input, 'utf8'));
  const rows = makeRows(raw, opts);

  const meta = {
    createdAt: new Date().toISOString(),
    input: path.resolve(opts.input),
    eligibleCount: rows.length,
    uniqueHostCount: new Set(rows.map(r => r.rawHost || '__missing_host__')).size,
    options: opts,
  };
  await fs.writeFile(path.join(opts.outdir, 'run_meta.json'), JSON.stringify(meta, null, 2), 'utf8');

  const context = await chromium.launchPersistentContext(opts.profile, {
    headless: !opts.headed,
    acceptDownloads: true,
    viewport: { width: 1440, height: 1100 },
  });

  const results = [];
  try {
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      console.log(`[${i + 1}/${rows.length}] ${row.rctId} :: ${row.rawHost} :: ${row.sourceUrl}`);
      try {
        const result = await scanOne(context, row, opts, artifactDirs);
        results.push(result);
        console.log(`  final=${result.finalFamily} | navOk=${result.navigationOk} | candidates=${(result.candidates || []).length} | loginLikely=${result.loginLikely}`);
      } catch (err) {
        console.error(`  failed: ${err.message}`);
        results.push({
          rowId: row.rowId,
          rctId: row.rctId,
          title: row.title,
          sourceUrl: row.sourceUrl,
          rawHost: row.rawHost,
          initialFamily: row.initialFamily,
          initialTier: row.initialTier,
          finalFamily: row.initialFamily,
          finalTier: row.initialTier,
          navigationOk: false,
          navigationError: err.message,
          fileLikeLinks: [],
          forms: [],
          candidates: [],
          networkHints: [],
          directDownloads: [],
          popupUrls: [],
          dialogs: [],
          probed: [],
        });
      } finally {
        await writeOutputs(opts.outdir, results);
      }
    }
  } finally {
    await context.close().catch(() => {});
  }

  console.log('\nSummary by resolved family:');
  for (const row of summarize(results)) {
    console.log(`${row.family.padEnd(22)} count=${String(row.count).padStart(3)} navOk=${String(row.navOk).padStart(3)} login=${String(row.loginLikely).padStart(3)} candidates=${String(row.hasCandidates).padStart(3)}`);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
