"""Portable JSON, Markdown, and escaped static HTML reports."""
from pathlib import Path
import html

from training_parity.common import write_new


def failure_details(row):
    details = []
    for gate in row.get("comparison", {}).get("gates", []):
        if not gate["passed"]:
            for name, check in gate["checks"].items():
                if not check["passed"]:
                    tensor = row["comparison"]["comparisons"][name][str(gate["step"])]["tensors"].get(gate["unit"], {})
                    details.append({"step": gate["step"], "tensor_or_group": gate["unit"],
                                    "rule": gate["rule"], "reasons": check["reasons"],
                                    "metrics": check["metrics"], "max_abs_location": tensor.get("max_abs_location")})
    return sorted(details, key=lambda d: (d["step"], d["tensor_or_group"]))


def markdown(report):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    counts = report["counts"]
    lines = [f"# {safe(report['name'])}", "", f"Status: **{report['status']}**", "",
             f"Requested: {counts['requested']}; required: {counts['required']}; "
             f"completed: {counts['completed']}; passed: {counts['passed']}.", "",
             "| Case | Configuration | Execution | Validation | Reason |",
             "| --- | --- | --- | --- | --- |"]
    for row in report["rows"]:
        reason = row.get("reason", "; ".join(row["reasons"]))
        failures = failure_details(row)
        if failures:
            first = failures[0]
            reason = f"step {first['step']}: {first['tensor_or_group']} — " + "; ".join(first["reasons"])
        lines.append("| " + " | ".join(map(safe, [row["case"], row["id"], row["execution"], row["validation"], reason])) + " |")
    lines += ["", "## Scope", "", *[f"- {safe(x)}" for x in report["limitations"]], "",
              "See report.json for full per-tensor metrics, failure coordinates, policy and execution evidence.", ""]
    return "\n".join(lines)


def save_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "report.json", report)
    text = markdown(report)
    (output / "report.md").write_text(text, encoding="utf-8")
    title = html.escape(report["name"])
    rows = []
    for row in report["rows"]:
        reason = row.get("reason", "; ".join(row["reasons"]))
        failures = failure_details(row)
        if failures:
            first = failures[0]
            reason = f"step {first['step']}: {first['tensor_or_group']}: " + "; ".join(first["reasons"])
            if first["max_abs_location"]:
                location = first["max_abs_location"]
                reason += f"; largest error at {location['index']}, ref={location['reference']}, candidate={location['candidate']}"
        cells = [row["case"], row["id"], row["execution"], row["validation"], reason]
        rows.append('<tr>' + ''.join('<td>'+html.escape(str(x))+'</td>' for x in cells) + '</tr>')
    counts = report["counts"]
    summary = f"Requested {counts['requested']} · Required {counts['required']} · Completed {counts['completed']} · Passed {counts['passed']}"
    limits = ''.join('<li>'+html.escape(x)+'</li>' for x in report["limitations"])
    (output / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{title}</title><style>body{{font:15px system-ui;margin:2rem;max-width:1500px;color:#192836}}'
        'table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.7rem;border-bottom:1px solid #d2dce3;vertical-align:top}'
        'th{background:#eef3f6}td{overflow-wrap:anywhere}li{margin:.5rem 0}.scroll{overflow:auto}a{color:#17607c}</style>'
        f'<h1>{title}</h1><p>Status: <strong>{html.escape(report["status"])}</strong></p>'
        f'<p>{summary}</p><p><a href="report.json">Full JSON evidence</a> · <a href="report.md">Markdown</a></p>'
        '<div class="scroll"><table><thead><tr><th>Case</th><th>Configuration</th><th>Execution</th>'
        '<th>Validation</th><th>Detail</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
        f'<h2>Scope</h2><ul>{limits}</ul></html>', encoding="utf-8")
    return output
