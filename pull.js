#!/usr/bin/env node
// pull.js — fetch full OHLCV history for a symbol/timeframe from the live
// TradingView Desktop app (via the tradingview-mcp CDP connection) and write CSV.
//
// Usage: node pull.js <symbol> <timeframe> <outcsv>
//   symbol:    e.g. OANDA:EURUSD
//   timeframe: e.g. D (daily), 240 (4H)
//   outcsv:    path to write the CSV (data/<name>.csv)
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
import { writeFileSync, mkdirSync } from 'node:fs';

const mcp = 'C:/Users/musuk/Desktop/TradingView/tradingview-mcp/src';
const { evaluate, connect, disconnect } = await import(pathToFileURL(resolve(mcp, 'connection.js')).href);
const { setSymbol, setTimeframe, setVisibleRange } = await import(pathToFileURL(resolve(mcp, 'core/chart.js')).href);

const [, , symbolArg, tfArg, outCsv] = process.argv;
if (!symbolArg || !tfArg || !outCsv) {
  console.error('Usage: node pull.js <symbol> <timeframe> <outcsv>');
  process.exit(1);
}

const FULL_SERIES_JS = `
  (function() {
    var bars = window.TradingViewApi._activeChartWidgetWV.value()._chartWidget.model().mainSeries().bars();
    if (!bars || typeof bars.lastIndex !== 'function') return null;
    var end = bars.lastIndex();
    var start = bars.firstIndex();
    var result = [];
    for (var i = start; i <= end; i++) {
      var v = bars.valueAt(i);
      if (v) result.push({ time: v[0], open: v[1], high: v[2], low: v[3], close: v[4], volume: v[5] || 0 });
    }
    return result;
  })()
`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  await connect();
  console.log(`Setting symbol=${symbolArg} timeframe=${tfArg}`);
  await setSymbol({ symbol: symbolArg });
  await sleep(2500);
  await setTimeframe({ timeframe: tfArg });
  await sleep(1500);

  // Pull the visible history plus paged-back history so the series is fully loaded.
  const now = Math.floor(Date.now() / 1000);
  const startEpoch = 946684800; // 2000-01-01
  console.log('Widening range to force history load...');
  await setVisibleRange({ from: startEpoch, to: now });
  await sleep(8000);

  const bars = await evaluate(FULL_SERIES_JS);
  if (!bars || bars.length === 0) throw new Error('No bars returned. Is the chart loaded?');
  bars.sort((a, b) => a.time - b.time);

  const dir = dirname(resolve(outCsv));
  mkdirSync(dir, { recursive: true });
  const header = 'time,open,high,low,close,volume';
  const rows = bars.map((b) => `${b.time},${b.open},${b.high},${b.low},${b.close},${b.volume}`);
  writeFileSync(resolve(outCsv), [header, ...rows].join('\n'), 'utf8');

  const first = new Date(bars[0].time * 1000).toISOString().slice(0, 10);
  const last = new Date(bars[bars.length - 1].time * 1000).toISOString().slice(0, 10);
  console.log(`Wrote ${bars.length} bars to ${outCsv}`);
  console.log(`  range: ${first} .. ${last}`);
  await disconnect();
}

main().catch((err) => {
  console.error('FAILED:', err.message);
  process.exit(1);
});