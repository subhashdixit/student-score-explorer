import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  predict,
  scoreRecord,
  parseCSV,
  csvText,
  sigmoid,
} from "../site/model-engine.js";
const model = JSON.parse(fs.readFileSync("site/model.json", "utf8"));
const sample = JSON.parse(fs.readFileSync("site/sample.json", "utf8"));
const fixtures = JSON.parse(
  fs.readFileSync("reports/parity-fixtures.json", "utf8"),
);
test("portable browser scores match real held-out Python model predictions", () => {
  for (const row of fixtures)
    assert.ok(
      Math.abs(scoreRecord(model, row.record) - row.expected) < 1e-7,
      `Mismatch ${scoreRecord(model, row.record)} vs ${row.expected}`,
    );
});
test("known input produces finite prediction and allows explicit training imputation", () => {
  assert.ok(Number.isFinite(scoreRecord(model, sample)));
  const partial = { ...sample, [model.features[0].name]: null };
  assert.ok(Number.isFinite(scoreRecord(model, partial)));
});
test("missing fields, all-null profiles and unbounded batches fail", () => {
  for (const rows of [[], [{}], Array(501).fill(sample), null])
    assert.throws(() => predict(model, rows));
  assert.throws(() =>
    predict(model, [
      Object.fromEntries(model.features.map((f) => [f.name, null])),
    ]),
  );
});
test("invalid numeric input and unsupported categories cannot silently predict", () => {
  const f = model.features.find((f) => f.type === "number");
  for (const value of [true, "NaN", Infinity, f.max * 2 + 1])
    assert.throws(() => scoreRecord(model, { ...sample, [f.name]: value }));
  const c = model.features.find((f) => f.type === "category");
  if (c)
    assert.throws(() =>
      scoreRecord(model, { ...sample, [c.name]: "not-a-valid-category" }),
    );
});
test("threshold validation and stable sigmoid cover operating extremes", () => {
  for (const threshold of [true, 0, 1, "NaN", Infinity])
    assert.throws(() => predict(model, [sample], threshold));
  assert.equal(sigmoid(10000), 1);
  assert.equal(sigmoid(-10000), 0);
  assert.equal(sigmoid(0), 0.5);
  if (model.kind === "classification") {
    assert.equal(
      predict(model, [sample], 0.01)[0].label,
      scoreRecord(model, sample) >= 0.01 ? 1 : 0,
    );
  }
});
test("CSV parser handles BOM, quoted commas, multiline values and CRLF", () => {
  assert.deepEqual(parseCSV('\uFEFFa,b\r\n"x,y","one\ntwo"\r\n'), [
    { a: "x,y", b: "one\ntwo" },
  ]);
  for (const csv of ["a,a\n1,2", "a,b\n1", 'a\n"open', 'a\n"x"bad'])
    assert.throws(() => parseCSV(csv));
});
test("CSV round-trip preserves synthetic sample and exports reject spreadsheet formulas", () => {
  const rows = parseCSV(csvText([sample]));
  assert.ok(Number.isFinite(scoreRecord(model, rows[0])));
  assert.ok(csvText([{ text: '=HYPERLINK("bad")' }]).includes("'=HYPERLINK"));
});
test("model hash and held-out split metadata identify the actual release", async () => {
  const crypto = await import("node:crypto");
  const report = JSON.parse(fs.readFileSync("site/metrics.json"));
  assert.equal(
    crypto
      .createHash("sha256")
      .update(fs.readFileSync("site/model.json"))
      .digest("hex"),
    report.artifact_sha256,
  );
  assert.ok(report.split.train > report.split.test);
  assert.equal(report.test.rows, report.split.test);
  assert.equal(report.training_only_preprocessing, true);
});
