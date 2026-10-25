#!/usr/bin/env node
/** Real-browser regression checks. Requires Playwright and a Streamlit runtime.
 * Run: node scripts/browser_checks.cjs --python .venv/Scripts/python.exe --out <fresh-dir>
 * Set GUARDFACE_BROWSER_CHANNEL=chrome to use an installed Chrome locally.
 * Add --case <name-fragment> to repeat one check; zero matches is an error.
 * Inputs are synthetic or provenance-documented public fixtures. No camera is
 * activated and fixture smoke tests do not establish biometric accuracy.
 */
'use strict';

const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');
const { chromium } = require('playwright');

const options = Object.fromEntries(process.argv.slice(2).reduce((pairs, item, index, all) => {
  if (item.startsWith('--')) pairs.push([item.slice(2), all[index + 1]]);
  return pairs;
}, []));
const root = path.resolve(__dirname, '..');
const output = path.resolve(options.out || path.join(require('node:os').tmpdir(), `guardface-browser-${Date.now()}`));
fs.mkdirSync(output, { recursive: true });
const results = [];
let server;
let browser;
let baseURL = options['base-url'];
const browserErrors = [];

async function waitUntil(check, message, timeout = 30000) {
  const deadline = Date.now() + timeout;
  let lastError;
  while (Date.now() < deadline) {
    try { if (await check()) return; } catch (error) { lastError = error; }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`${message}${lastError ? `: ${lastError.message}` : ''}`);
}

async function freePort() {
  const listener = net.createServer();
  await new Promise((resolve, reject) => {
    listener.once('error', reject);
    listener.listen(0, '127.0.0.1', resolve);
  });
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  return port;
}

async function startServer() {
  if (baseURL) {
    const parsed = new URL(baseURL);
    assert(['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname), 'Only localhost targets are allowed');
    return;
  }
  const port = await freePort();
  baseURL = `http://127.0.0.1:${port}`;
  const log = fs.createWriteStream(path.join(output, 'server.log'));
  server = spawn(options.python || 'python', ['-B', '-m', 'streamlit', 'run', path.join(root, 'app_demo.py'),
    '--server.address', '127.0.0.1', '--server.port', String(port), '--server.headless', 'true',
    '--browser.gatherUsageStats', 'false'], {
    cwd: root, windowsHide: true,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', STREAMLIT_BROWSER_GATHER_USAGE_STATS: 'false' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.stdout.pipe(log);
  server.stderr.pipe(log);
  let launchError;
  server.on('error', error => { launchError = error; });
  await waitUntil(async () => {
    if (launchError) throw launchError;
    if (server.exitCode !== null) throw new Error(`Streamlit exited ${server.exitCode}; see server.log`);
    return (await fetch(`${baseURL}/_stcore/health`, { signal: AbortSignal.timeout(3000) })).ok;
  }, 'Local Streamlit server did not become healthy', 60000);
}

async function settle(page) {
  await page.getByRole('heading', { name: 'GuardFace Research Demo' }).waitFor();
  let stableSince = Date.now();
  await waitUntil(async () => {
    if (await page.getByRole('button', { name: 'Stop', exact: true }).count()) stableSince = Date.now();
    return Date.now() - stableSince >= 300;
  },
    'Streamlit rerun did not finish');
  assert.equal(await page.locator('[data-testid="stException"]').count(), 0, 'No Streamlit exception');
}

async function openPage(contextOptions = {}) {
  const context = await browser.newContext(contextOptions);
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  page.on('pageerror', error => browserErrors.push(error.message));
  await page.goto(baseURL);
  await page.getByRole('button', { name: 'Run template demonstration', exact: true }).waitFor({ timeout: 60000 });
  await settle(page);
  return { page, context };
}

async function safeDiagnostic(page) {
  const text = await page.locator('body').innerText();
  assert(!text.includes('IDENTITY VERIFIED'), 'The browser must not authenticate');
  assert.equal(await page.locator('[data-testid="stAlertContainer"]:has-text("Identity verified")').count(), 0);
  assert(text.includes('Research diagnostics only.'), 'Research warning stays visible');
}

async function chooseInput(page, group, choice) {
  await page.getByRole('radiogroup', { name: group, exact: true }).getByText(choice, { exact: true }).click();
  await waitUntil(async () => await page.getByRole('radiogroup', { name: group, exact: true })
    .getByRole('radio', { name: choice, exact: true }).isChecked(), 'Radio input selection');
  await settle(page);
}

async function screenshot(page, name) {
  await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: true });
}

async function displayedCosine(page, label) {
  const metric = page.locator('[data-testid="stMetric"]').filter({ hasText: label });
  assert.equal(await metric.count(), 1, `One ${label} metric`);
  const match = (await metric.innerText()).match(/-?\d+\.\d{4}/);
  assert(match, `A numeric ${label} value is displayed`);
  const value = Number(match[0]);
  assert(Number.isFinite(value) && value >= -1 && value <= 1, 'Cosine stays finite and bounded');
  return value;
}

function publicFaceFixtures() {
  const reference = path.join(root, 'tests', 'fixtures', 'astronaut.png');
  const different = path.resolve(options['other-fixture'] || path.join(root, 'tests', 'fixtures', 'grace_hopper.jpg'));
  for (const fixture of [reference, different]) assert(fs.existsSync(fixture), `Public fixture exists: ${path.basename(fixture)}`);
  const variant = path.join(output, 'astronaut-brightness-rotation.png');
  const transform = [
    'from PIL import Image, ImageEnhance',
    'import sys',
    'with Image.open(sys.argv[1]) as source:',
    '    rgb = ImageEnhance.Brightness(source.convert("RGB")).enhance(0.93)',
    '    rgb.rotate(2, resample=Image.Resampling.BICUBIC).save(sys.argv[2], format="PNG")',
  ].join('\n');
  const result = spawnSync(options.python || 'python', ['-B', '-c', transform, reference, variant], {
    cwd: root, windowsHide: true, encoding: 'utf8',
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
  });
  assert.equal(result.status, 0, `Public-fixture transform succeeds: ${result.stderr || result.error || ''}`);
  const sha256 = fixture => createHash('sha256').update(fs.readFileSync(fixture)).digest('hex');
  return { reference, variant, different, sha256: { reference: sha256(reference), variant: sha256(variant), different: sha256(different) } };
}

async function diagnosticContrast(page) {
  const observations = await page.locator('[data-testid="stMarkdownContainer"] p').evaluateAll(elements => {
    const labels = ['Moiré score:', 'High frequency ratio:', 'LBP entropy:', 'Chrominance diversity:'];
    const luminance = rgb => {
      const c = rgb.map(value => value / 255).map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
      return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
    };
    return elements.filter(element => labels.some(label => element.textContent.startsWith(label))).map(element => {
      let ancestor = element;
      while (ancestor && getComputedStyle(ancestor).backgroundColor === 'rgba(0, 0, 0, 0)') ancestor = ancestor.parentElement;
      const foreground = getComputedStyle(element).color;
      const background = ancestor ? getComputedStyle(ancestor).backgroundColor : 'rgb(255, 255, 255)';
      const fg = foreground.match(/[\d.]+/g).map(Number);
      const bg = background.match(/[\d.]+/g).map(Number);
      const first = luminance(fg), second = luminance(bg);
      return { label: element.textContent.split(':')[0], foreground, background,
        ratio: (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05) };
    });
  });
  assert.equal(observations.length, 4, 'All four diagnostic values render as labelled plain text');
  assert(observations.every(value => value.ratio >= 4.5), `Diagnostic text contrast: ${JSON.stringify(observations)}`);
  return observations;
}

async function run(name, check) {
  if (options.case && !name.includes(options.case)) return;
  const started = performance.now();
  try {
    const detail = await check();
    results.push({ name, status: 'passed', milliseconds: Math.round(performance.now() - started), detail });
  } catch (error) {
    results.push({ name, status: 'failed', milliseconds: Math.round(performance.now() - started), error: error.message });
  }
  console.log(`${results.at(-1).status}: ${name}`);
  if (results.at(-1).error) console.error(results.at(-1).error);
}

async function layoutCheck(width, height, theme) {
  const { page, context } = await openPage({ viewport: { width, height }, colorScheme: theme });
  try {
    const overflow = await page.evaluate(() => ({
      viewport: document.documentElement.clientWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
    }));
    assert(overflow.document <= overflow.viewport + 1 && overflow.body <= overflow.viewport + 1,
      `Horizontal overflow: ${JSON.stringify(overflow)}`);
    await safeDiagnostic(page);
    const diagnosticText = await diagnosticContrast(page);
    for (const tab of ['PAD & Template Inspector', 'Pair Comparison', 'Synthetic Benchmark']) {
      await page.getByRole('tab', { name: new RegExp(tab.replace('&', '\\&')) }).click();
      assert(await page.getByRole('tab', { selected: true }).innerText(), 'Tabs remain reachable');
    }
    await page.getByRole('tab').first().click();
    await screenshot(page, `layout-${width}-${theme}`);
    await page.getByRole('button', { name: 'Run template demonstration', exact: true }).scrollIntoViewIfNeeded();
    await screenshot(page, `layout-${width}-${theme}-operation`);
    return { ...overflow, theme, diagnosticText };
  } finally { await context.close(); }
}

async function main() {
  await startServer();
  browser = await chromium.launch({ headless: true,
    ...(process.env.GUARDFACE_BROWSER_CHANNEL ? { channel: process.env.GUARDFACE_BROWSER_CHANNEL } : {}),
  });
  for (const [width, height, theme] of [[1440, 1050, 'light'], [390, 844, 'light'], [390, 844, 'dark']]) {
    await run(`responsive ${width}px ${theme}`, () => layoutCheck(width, height, theme));
  }
  await run('labelled controls and keyboard tab navigation', async () => {
    const { page, context } = await openPage({ viewport: { width: 1440, height: 1050 } });
    try {
      for (const label of ['PAD heuristic threshold', 'SLERP rotation alpha', 'Feature dropout rate', 'Experimental cosine threshold']) {
        // React Aria currently repeats the visible label in the accessible name.
        assert.equal(await page.getByRole('slider', { name: new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`) }).count(), 1);
      }
      assert.equal(await page.getByText('0.363', { exact: true }).count(), 1, 'Configured threshold is displayed with three-decimal precision');
      assert.equal(await page.getByRole('checkbox', { name: 'Use synthetic feature demo', exact: true }).count(), 1);
      assert.equal(await page.getByRole('checkbox', { name: 'Enforce Zero-Leakage Protocol', exact: true }).count(), 1);
      assert.equal(await page.getByRole('textbox', { name: 'Trained checkpoint path (optional)', exact: true }).count(), 1);
      assert.equal(await page.getByRole('radiogroup', { name: 'Inspector input:', exact: true }).count(), 1);
      const firstTab = page.getByRole('tab').first();
      await firstTab.focus();
      await page.keyboard.press('ArrowRight');
      await page.keyboard.press('Enter');
      await waitUntil(async () => (await page.getByRole('tab', { selected: true }).innerText()).includes('Pair Comparison'), 'Keyboard tab selection');
      await safeDiagnostic(page);
      return 'All research configuration controls and the inspector group have accessible names; tabs activate with keyboard.';
    } finally { await context.close(); }
  });
  await run('empty uploads and cameras disable actions', async () => {
    const { page, context } = await openPage({ viewport: { width: 1440, height: 1050 } });
    try {
      await chooseInput(page, 'Inspector input:', 'Upload');
      await waitUntil(() => page.getByRole('button', { name: 'Run template demonstration', exact: true }).isDisabled(), 'Empty upload disables inspector');
      await chooseInput(page, 'Inspector input:', 'Camera');
      await waitUntil(() => page.getByRole('button', { name: 'Run template demonstration', exact: true }).isDisabled(), 'Empty camera disables inspector');
      // No permission is granted and no camera button is activated.
      await page.getByRole('tab', { name: /Pair Comparison/ }).click();
      for (const [first, second] of [['Upload', 'Upload'], ['Camera', 'Upload'], ['Preset', 'Camera']]) {
        await chooseInput(page, 'Image 1 input:', first);
        await chooseInput(page, 'Image 2 input:', second);
        await waitUntil(() => page.getByRole('button', { name: 'Run pair comparison', exact: true }).isDisabled(), 'Missing pair input disables comparison');
      }
      await safeDiagnostic(page);
      await screenshot(page, 'missing-inputs');
      return 'Empty upload/camera choices never substitute presets or enable processing.';
    } finally { await context.close(); }
  });
  await run('corrupt upload displays a recoverable error', async () => {
    const { page, context } = await openPage({ viewport: { width: 1440, height: 1050 } });
    try {
      await chooseInput(page, 'Inspector input:', 'Upload');
      await page.locator('input[type=file]').first().setInputFiles({ name: 'corrupt.png', mimeType: 'image/png', buffer: Buffer.from('not an image') });
      await page.getByText('Unable to read this image.', { exact: false }).waitFor();
      await settle(page);
      assert(await page.getByRole('button', { name: 'Run template demonstration', exact: true }).isDisabled());
      await safeDiagnostic(page);
      await screenshot(page, 'corrupt-upload');
      return 'Invalid PNG produces the labelled input error, with action disabled and no crash.';
    } finally { await context.close(); }
  });
  await run('non-face presets and explicit synthetic comparison stay unverified', async () => {
    const { page, context } = await openPage({ viewport: { width: 1440, height: 1050 } });
    try {
      await page.getByRole('button', { name: 'Run template demonstration', exact: true }).click();
      await page.getByText('No template demonstration completed.', { exact: false }).waitFor();
      await settle(page);
      await safeDiagnostic(page);
      await page.getByText('Use synthetic feature demo', { exact: true }).click();
      await settle(page);
      await page.getByRole('tab', { name: /Pair Comparison/ }).click();
      await page.getByRole('button', { name: 'Run pair comparison', exact: true }).click();
      await page.getByText('Processing status: DEMO_ONLY', { exact: true }).waitFor();
      await settle(page);
      await page.getByText('Identity remains unverified;', { exact: false }).waitFor();
      await safeDiagnostic(page);
      assert.equal(await page.locator('[data-testid="stMetric"]').filter({ hasText: 'descriptor cosine' }).count(), 2);
      await screenshot(page, 'pair-demo-unverified');
      return 'Generated textures are rejected by the real-face path; explicit statistics demo returns only diagnostic cosines.';
    } finally { await context.close(); }
  });
  await run('parallel browser sessions keep input mode and demo selection isolated', async () => {
    const sessions = await Promise.all([openPage(), openPage(), openPage()]);
    try {
      await chooseInput(sessions[0].page, 'Inspector input:', 'Upload');
      await sessions[1].page.getByText('Use synthetic feature demo', { exact: true }).click();
      await settle(sessions[1].page);
      await waitUntil(() => sessions[0].page.getByRole('button', { name: 'Run template demonstration', exact: true }).isDisabled(), 'First session retains empty upload');
      assert(!(await sessions[1].page.getByRole('button', { name: 'Run template demonstration', exact: true }).isDisabled()));
      assert(!(await sessions[2].page.getByRole('checkbox', { name: 'Use synthetic feature demo', exact: true }).isChecked()));
      await Promise.all(sessions.map(({ page }) => safeDiagnostic(page)));
      return 'Three independent contexts preserve their own input mode and feature-demo state.';
    } finally { await Promise.all(sessions.map(({ context }) => context.close())); }
  });
  await run('synthetic benchmark renders measured research-only results', async () => {
    const { page, context } = await openPage();
    try {
      await page.getByRole('tab', { name: /Synthetic Benchmark/ }).click();
      await page.getByRole('button', { name: 'Run synthetic benchmark', exact: true }).click();
      await page.getByText('Evaluated 105 generated patterns.', { exact: false }).waitFor();
      await settle(page);
      for (const metric of ['Synthetic BPCER', 'Synthetic APCER', 'Synthetic ACER']) {
        assert.equal(await page.locator('[data-testid="stMetric"]').filter({ hasText: metric }).count(), 1);
      }
      await safeDiagnostic(page);
      await screenshot(page, 'synthetic-benchmark');
      return '105 generated inputs produce the three labelled synthetic error rates and retained validation caveat.';
    } finally { await context.close(); }
  });
  await run('pretrained public-face uploads compare and retain accurate session results', async () => {
    const fixtures = publicFaceFixtures();
    const { page, context } = await openPage({ viewport: { width: 1440, height: 1050 } });
    try {
      await page.getByText('Face backend ready:', { exact: false }).waitFor();
      assert(!(await page.getByRole('checkbox', { name: 'Use synthetic feature demo', exact: true }).isChecked()));
      // Exercise advisory face comparison separately from the optional strict
      // heuristic gate introduced by the dual-tier extension.
      const strictGate = page.getByRole('checkbox', { name: 'Enforce Zero-Leakage Protocol', exact: true });
      if (await strictGate.isChecked()) {
        await page.getByText('Enforce Zero-Leakage Protocol', { exact: true }).click();
        await settle(page);
      }
      assert(!(await strictGate.isChecked()), 'Advisory comparison mode is explicitly selected');
      await page.getByRole('tab', { name: /Pair Comparison/ }).click();
      await chooseInput(page, 'Image 1 input:', 'Upload');
      await chooseInput(page, 'Image 2 input:', 'Upload');
      const referenceUpload = page.getByTestId('stFileUploader').filter({ hasText: 'Upload Image 1' }).locator('input[type=file]');
      const comparisonUpload = page.getByTestId('stFileUploader').filter({ hasText: 'Upload Image 2' }).locator('input[type=file]');
      assert.equal(await referenceUpload.count(), 1, 'Named reference upload widget');
      assert.equal(await comparisonUpload.count(), 1, 'Named comparison upload widget');
      await referenceUpload.setInputFiles(fixtures.reference);
      await settle(page);
      await comparisonUpload.setInputFiles(fixtures.variant);
      await settle(page);
      await page.getByRole('button', { name: 'Run pair comparison', exact: true }).click();
      await page.getByText('Processing status: RESEARCH_COMPARISON', { exact: true }).waitFor({ timeout: 120000 });
      await settle(page);
      await page.getByText('Experimental comparison: similar.', { exact: false }).waitFor();
      const sameSourceCosine = await displayedCosine(page, 'Raw descriptor cosine (diagnostic)');
      const transformedCosine = await displayedCosine(page, 'Transformed descriptor cosine (diagnostic)');
      assert(sameSourceCosine > 0.393, 'Mildly transformed source clears the default upper uncertainty bound');
      assert.equal(await page.getByText('presentation analysis: MODEL_ESTIMATE.', { exact: false }).count(), 2,
        'Both real inputs receive learned advisory presentation estimates');
      assert.equal(await page.getByText('Bona fide model score (uncalibrated):', { exact: false }).count(), 2);
      await safeDiagnostic(page);
      await page.getByText('Experimental comparison: similar.', { exact: false }).scrollIntoViewIfNeeded();
      await screenshot(page, 'real-pair-same-source');
      await page.locator('[data-testid="stMetric"]').filter({ hasText: 'Raw descriptor cosine (diagnostic)' }).scrollIntoViewIfNeeded();
      await screenshot(page, 'real-pair-same-source-scores');
      // Changing only the inspector triggers a rerun without changing pair inputs.
      await page.getByRole('tab').first().click();
      await chooseInput(page, 'Inspector input:', 'Upload');
      await page.getByRole('tab', { name: /Pair Comparison/ }).click();
      await page.getByText('Processing status: RESEARCH_COMPARISON', { exact: true }).waitFor();
      assert.equal(await displayedCosine(page, 'Raw descriptor cosine (diagnostic)'), sameSourceCosine,
        'Pair result survives an unrelated inspector rerun');
      // A new paired image invalidates the old result before another operation.
      await comparisonUpload.setInputFiles(fixtures.different);
      await settle(page);
      await waitUntil(async () => await page.getByText('Processing status: RESEARCH_COMPARISON', { exact: true }).count() === 0,
        'Changed pair image removes stale results');
      await page.getByRole('button', { name: 'Run pair comparison', exact: true }).click();
      await page.getByText('Processing status: RESEARCH_COMPARISON', { exact: true }).waitFor({ timeout: 120000 });
      await settle(page);
      await page.getByText('Experimental comparison: different.', { exact: false }).waitFor();
      const differentSourceCosine = await displayedCosine(page, 'Raw descriptor cosine (diagnostic)');
      assert(differentSourceCosine < 0.333, 'Separate-source face is below the lower uncertainty bound');
      await safeDiagnostic(page);
      await page.getByText('Experimental comparison: different.', { exact: false }).scrollIntoViewIfNeeded();
      await screenshot(page, 'real-pair-different-source');
      await page.locator('[data-testid="stMetric"]').filter({ hasText: 'Raw descriptor cosine (diagnostic)' }).scrollIntoViewIfNeeded();
      await screenshot(page, 'real-pair-different-source-scores');
      const narrowLayouts = [];
      await page.setViewportSize({ width: 390, height: 844 });
      await waitUntil(async () => {
        const sidebar = page.getByTestId('stSidebar');
        if (await sidebar.count() === 0) return true;
        const box = await sidebar.boundingBox();
        return box === null || box.x + box.width <= 1;
      }, 'Responsive sidebar finishes collapsing before screenshots');
      for (const theme of ['light', 'dark']) {
        await page.emulateMedia({ colorScheme: theme });
        await settle(page);
        await page.getByText('Experimental comparison: different.', { exact: false }).scrollIntoViewIfNeeded();
        const bounds = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth,
          document: document.documentElement.scrollWidth, body: document.body.scrollWidth }));
        assert(bounds.document <= bounds.viewport + 1 && bounds.body <= bounds.viewport + 1,
          `Real comparison has no narrow overflow: ${JSON.stringify(bounds)}`);
        await screenshot(page, `real-pair-390-${theme}`);
        narrowLayouts.push({ theme, ...bounds });
      }
      return { sameSourceCosine, transformedCosine, differentSourceCosine, fixtureSHA256: fixtures.sha256,
        narrowLayouts,
        interpretation: 'Two public photographs plus one same-source transformation exercise real pretrained inference, uploads and session state. This is not independent identity or physical-PAD validation.' };
    } catch (error) {
      await screenshot(page, 'real-comparison-failure');
      fs.writeFileSync(path.join(output, 'real-comparison-failure.txt'), await page.locator('body').innerText());
      throw error;
    } finally { await context.close(); }
  });
  await run('configured heuristic gate reports its rejected operation', async () => {
    const fixtures = publicFaceFixtures();
    const { page, context } = await openPage({ viewport: { width: 1440, height: 1050 } });
    try {
      const strictGate = page.getByRole('checkbox', { name: 'Enforce Zero-Leakage Protocol', exact: true });
      assert(await strictGate.isChecked(), 'The configured strict policy remains enabled by default');
      await page.getByRole('tab', { name: /Pair Comparison/ }).click();
      await chooseInput(page, 'Image 1 input:', 'Upload');
      await chooseInput(page, 'Image 2 input:', 'Upload');
      for (const label of ['Upload Image 1', 'Upload Image 2']) {
        await page.getByTestId('stFileUploader').filter({ hasText: label }).locator('input[type=file]').setInputFiles(fixtures.different);
        await settle(page);
      }
      await page.getByRole('button', { name: 'Run pair comparison', exact: true }).click();
      await page.getByText('Processing status: ZERO_LEAKAGE_TRIGGERED', { exact: true }).waitFor({ timeout: 120000 });
      await settle(page);
      assert.equal(await page.locator('[data-testid="stMetric"]').filter({ hasText: 'Raw descriptor cosine (diagnostic)' }).count(), 0);
      await safeDiagnostic(page);
      await screenshot(page, 'configured-gate-rejection');
      return 'Configured heuristic rejection prevents a comparison result. A public photo is only a functional fixture; the test does not establish physical spoof resistance or absence of intermediate feature computation.';
    } finally { await context.close(); }
  });
  assert.equal(browserErrors.length, 0, `Browser errors: ${browserErrors.join('; ')}`);
  assert(results.length > 0, 'No browser checks matched the selected case');
}

main().catch(error => {
  results.push({ name: 'browser infrastructure', status: 'failed', error: error.message });
}).finally(async () => {
  const browserVersion = browser ? browser.version() : null;
  try { await browser?.close(); } finally {
    if (server && server.exitCode === null) {
      // Windows venv launchers may own another Python process: stop only our tree.
      if (process.platform === 'win32') {
        spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
      } else server.kill();
    }
  }
  const summary = { generatedUTC: new Date().toISOString(), browserVersion, baseURL, results, browserErrors,
    passed: results.filter(result => result.status === 'passed').length,
    failed: results.filter(result => result.status === 'failed').length };
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  if (summary.failed) process.exitCode = 1;
});
