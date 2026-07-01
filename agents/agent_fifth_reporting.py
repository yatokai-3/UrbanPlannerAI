""" Agent 5: Report Writer — packages the whole pipeline into a full report.

The "pharmacist": it doesn't decide, it presents. It pulls from EVERY upstream
agent to tell the complete story:
  Agent 1 (data foundation) -> Agent 2 (demand analysis) -> Agent 3 (solutions,
  with the deterministic cost/ridership tables) -> Agent 4 (independent review).

Design principle (same as the rest of the pipeline): the NUMBERS and tables are
rendered straight from the agent JSON (accurate, no re-hallucination); only the
executive summary + conclusion prose is LLM-written (one cheap call).

Output: a polished PDF (via fpdf2) plus a Markdown fallback that always works.
"""

import json
import os
import time
import re
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import config
from utils import token_meter

try:
    from fpdf import FPDF, XPos, YPos
    _FPDF_OK = True
except ImportError:
    _FPDF_OK = False


client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Palette
NAVY = (23, 42, 69)
BLUE = (41, 98, 168)
LIGHT = (232, 238, 247)
GREY = (110, 110, 110)
GREEN = (22, 120, 60)
ORANGE = (190, 110, 20)
RED = (170, 40, 40)


# ---------------------------------------------------------------------------
# LLM (narrative only) — one small call
# ---------------------------------------------------------------------------

def chat_json(messages: list, max_tokens: int) -> dict:
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.LLM_JSON_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            token_meter.record(response)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            msg = str(e)
            if ("rate_limit" in msg or "429" in msg) and attempt < config.LLM_MAX_RETRIES - 1:
                m = re.search(r"try again in ([\d.]+)s", msg)
                wait = float(m.group(1)) + 0.5 if m else config.LLM_REQUEST_DELAY * (attempt + 1)
                print(f"  [report] rate limited — waiting {wait:.1f}s (attempt {attempt + 1}/{config.LLM_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Groq call failed after all retries")


_NARRATIVE_PROMPT = """
    You are a technical writer producing the front-matter prose for a city's
    transport plan report, for a decision-maker audience.

    You are given the demand ANALYSIS, the approved PLAN, and the reviewer's
    VERDICT. Write only the connective prose — the report renders all tables and
    numbers itself, so do NOT list raw numbers; summarise meaning.

    Return ONLY valid JSON:
    {
        "title": "report title naming the city",
        "executive_summary": "4-6 sentences a mayor could read: the problem, the
                              recommended direction, and the headline outcome",
        "approach_note": "2-3 sentences on the AI multi-agent method (data ->
                          analysis -> design with ridership/cost models -> review)",
        "conclusion": "3-4 sentence closing recommendation and next steps"
    }
"""


def _generate_narrative(analysis: dict, plan: dict, critique: dict) -> dict:
    # Strip the bulky raw evidence before sending — the PDF renders it itself.
    slim_plan = {k: v for k, v in (plan or {}).items() if k != "_computed_evidence"}
    print("  [report] writing narrative (1 LLM call)...")
    return chat_json(
        messages=[
            {"role": "system", "content": _NARRATIVE_PROMPT},
            {"role": "user", "content": json.dumps({"ANALYSIS": analysis, "PLAN": slim_plan, "VERDICT": critique or {}})},
        ],
        max_tokens=1500,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sanitize(text) -> str:
    """fpdf core fonts are latin-1; swap common glyphs, then hard-fallback."""
    s = str(text)
    repl = {"₹": "Rs ", "–": "-", "—": "-", "→": "->", "’": "'", "‘": "'",
            "“": '"', "”": '"', "•": "-", "≈": "~", "×": "x", "²": "2", "✓": "[ok]"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _fmt_value(v) -> str:
    """Flatten a JSON value (dict/list/scalar) into readable text."""
    if isinstance(v, dict):
        return "; ".join(f"{k.replace('_', ' ')}: {_fmt_value(val)}" for k, val in v.items())
    if isinstance(v, list):
        return "; ".join(_fmt_value(x) for x in v)
    return str(v)


# ---------------------------------------------------------------------------
# PDF document
# ---------------------------------------------------------------------------

if _FPDF_OK:
    class PlanPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return  # title page has no running header
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*GREY)
            self.cell(0, 8, _sanitize(self._doc_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
            self.set_draw_color(*LIGHT)
            self.line(self.l_margin, 16, self.w - self.r_margin, 16)
            self.ln(4)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*GREY)
            self.cell(0, 8, f"UrbanPlanner AI  |  Page {self.page_no()}", align="C")

        # --- building blocks ---
        def title_page(self, title, city, date_str):
            self.add_page()
            self.set_fill_color(*NAVY)
            self.rect(0, 0, self.w, 70, "F")
            self.set_y(28)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 24)
            self.multi_cell(0, 11, _sanitize(title), align="C")
            self.set_y(80)
            self.set_text_color(*NAVY)
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, _sanitize(f"City: {city}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*GREY)
            self.cell(0, 8, "AI-Generated Sustainable Transport Plan", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.cell(0, 8, _sanitize(date_str), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        def h1(self, text):
            if self.get_y() > self.h - 50:
                self.add_page()
            self.ln(3)
            self.set_fill_color(*NAVY)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 13)
            self.cell(0, 9, _sanitize(f"  {text}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            self.ln(2)

        def h2(self, text):
            self.ln(1)
            self.set_text_color(*BLUE)
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 7, _sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        def body(self, text):
            if not text:
                return
            self.set_x(self.l_margin)
            self.set_text_color(30, 30, 30)
            self.set_font("Helvetica", "", 10)
            self.multi_cell(self.epw, 5.5, _sanitize(text))
            self.ln(1.5)

        def bullets(self, items):
            self.set_text_color(30, 30, 30)
            self.set_font("Helvetica", "", 10)
            for it in items or []:
                self.set_x(self.l_margin)
                self.multi_cell(self.epw, 5.5, _sanitize(f"  -  {_fmt_value(it)}"))
            self.ln(1.5)

        def table(self, headers, rows, widths, verdict_col=None):
            self.set_font("Helvetica", "B", 8)
            self.set_fill_color(*BLUE)
            self.set_text_color(255, 255, 255)
            for h, w in zip(headers, widths):
                self.cell(w, 7, _sanitize(h), border=0, align="C", fill=True,
                          new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.ln(7)
            self.set_font("Helvetica", "", 8)
            fill = False
            for row in rows:
                self.set_fill_color(*(LIGHT if fill else (255, 255, 255)))
                for i, (val, w) in enumerate(zip(row, widths)):
                    if verdict_col is not None and i == verdict_col:
                        v = str(val).upper()
                        self.set_text_color(*(GREEN if "VIABLE" == v else RED if "NOT" in v else ORANGE))
                    else:
                        self.set_text_color(40, 40, 40)
                    self.cell(w, 6.5, _sanitize(val), border=0, align="C", fill=True,
                              new_x=XPos.RIGHT, new_y=YPos.TOP)
                self.ln(6.5)
                fill = not fill
            self.ln(3)


# ---------------------------------------------------------------------------
# Section builders (deterministic — pull straight from the agent JSON)
# ---------------------------------------------------------------------------

def _section_data_foundation(pdf, facts):
    if not facts:
        return
    sources = {f.get("title") for f in facts}
    n_facts = sum(len(f.get("key_facts", [])) for f in facts)
    pdf.h1("1. Data Foundation (Agent 1)")
    pdf.body(f"The plan is grounded in {n_facts} researched facts drawn from "
             f"{len(sources)} sources (Wikipedia, government mobility plans, traffic studies).")
    pdf.h2("Key sources")
    pdf.bullets(list(sources)[:8])


def _section_analysis(pdf, analysis):
    if not analysis:
        return
    pdf.h1("2. Demand Analysis (Agent 2)")
    km = analysis.get("key_metrics", {})
    if km:
        pdf.h2("Key metrics")
        pdf.bullets([f"{k.replace('_', ' ').title()}: {_fmt_value(v)}" for k, v in km.items()])
    for field, label in [("mobility_patterns", "Mobility patterns"),
                         ("current_demand", "Current demand"),
                         ("future_demand", "Future demand"),
                         ("capacity_gaps", "Capacity gaps"),
                         ("pt_deficiencies", "Public-transport deficiencies")]:
        if analysis.get(field):
            pdf.h2(label)
            pdf.body(_fmt_value(analysis[field]))
    if analysis.get("bottlenecks"):
        pdf.h2("Bottlenecks")
        pdf.bullets([_fmt_value(b) for b in analysis["bottlenecks"]])


def _section_solutions(pdf, plan):
    if not plan:
        return
    pdf.h1("3. Proposed Solutions (Agent 3)")
    # recommendation per corridor
    solutions = {s.get("corridor"): s for s in plan.get("corridor_solutions", [])}
    evidence = plan.get("_computed_evidence", [])

    headers = ["Mode", "Daily Rid.", "Peak/hr", "Capacity", "Capex (cr)", "Break-even", "Verdict"]
    widths = [24, 24, 22, 24, 26, 28, 42]

    for ev in evidence:
        name = ev.get("corridor", "Corridor")
        pdf.h2(name)
        meta = ev.get("inputs", {})
        pdf.body(f"{ev.get('origin','?')} -> {ev.get('destination','?')}  |  "
                 f"length {meta.get('route_length_km','?')} km, {meta.get('number_of_stops','?')} stops  "
                 f"(length: {ev.get('length_source','est.')})")
        rows = []
        for mode, e in ev.get("mode_evaluations", {}).items():
            be = e["break_even_years"]
            be_disp = "Never (subsidy)" if isinstance(be, str) else f"{be} yr"  # avoid long text overflow
            rows.append([mode.upper(), f"{e['daily_ridership']:,}", f"{e['peak_hour_ridership']:,}",
                         f"{e['peak_capacity_pphpd']:,}", e["capital_cost_crore"],
                         be_disp, e["verdict"]])
        pdf.table(headers, rows, widths, verdict_col=6)

        sol = solutions.get(name)
        if sol:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*GREEN)
            pdf.multi_cell(pdf.epw, 5.5, _sanitize(f"  Recommended: {str(sol.get('recommended_mode','')).upper()}"))
            pdf.body(f"Why: {sol.get('rationale','')}")
            if sol.get("alternatives_rejected"):
                pdf.body(f"Alternatives rejected: {sol.get('alternatives_rejected')}")

    if plan.get("citywide_recommendations"):
        pdf.h2("Citywide recommendations")
        pdf.body(_fmt_value(plan["citywide_recommendations"]))
    if plan.get("assumptions_and_caveats"):
        pdf.h2("Assumptions & caveats")
        pdf.bullets(plan["assumptions_and_caveats"])


def _section_review(pdf, critique):
    if not critique:
        return
    pdf.h1("4. Independent Review (Agent 4)")
    verdict = critique.get("overall_verdict", "")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*(GREEN if verdict == "APPROVED" else ORANGE))
    pdf.cell(0, 7, _sanitize(f"Verdict: {verdict}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.body(critique.get("overall_assessment", ""))
    if critique.get("key_weaknesses"):
        pdf.h2("Key weaknesses identified")
        pdf.bullets(critique["key_weaknesses"])
    if critique.get("improvement_suggestions"):
        pdf.h2("Improvement suggestions")
        pdf.bullets(critique["improvement_suggestions"])


# ---------------------------------------------------------------------------
# Markdown fallback (always available)
# ---------------------------------------------------------------------------

def _render_markdown(narrative, analysis, plan, critique, facts) -> str:
    L = [f"# {narrative.get('title','Urban Transport Plan')}", "",
         "## Executive Summary", narrative.get("executive_summary", ""), "",
         "## Approach", narrative.get("approach_note", ""), ""]
    if facts:
        L += ["## 1. Data Foundation",
              f"{sum(len(f.get('key_facts',[])) for f in facts)} facts from "
              f"{len({f.get('title') for f in facts})} sources.", ""]
    if analysis:
        L += ["## 2. Demand Analysis"]
        for k, v in (analysis.get("key_metrics", {}) or {}).items():
            L.append(f"- **{k.replace('_',' ').title()}:** {_fmt_value(v)}")
        L.append("")
    if plan:
        L += ["## 3. Proposed Solutions"]
        for s in plan.get("corridor_solutions", []):
            L.append(f"- **{s.get('corridor','')}** -> **{str(s.get('recommended_mode','')).upper()}** "
                     f"({s.get('rationale','')})")
        L.append("")
    if critique:
        L += ["## 4. Independent Review", f"**Verdict:** {critique.get('overall_verdict','')}",
              critique.get("overall_assessment", ""), ""]
    L += ["## Conclusion", narrative.get("conclusion", ""), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_report_agent(plan: dict, critique: dict = None, analysis: dict = None,
                     facts: list = None, out_basename: str = "urban_transport_plan") -> dict:
    """Write the full report from ALL agent outputs. Markdown always; PDF if fpdf2 present."""
    narrative = _generate_narrative(analysis, plan, critique)
    city = (analysis or {}).get("city") or (plan or {}).get("city") or "the city"
    title = narrative.get("title") or f"Sustainable Transport Plan for {city}"
    date_str = datetime.now().strftime("%d %B %Y")

    # Markdown (always)
    md_path = f"{out_basename}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(narrative, analysis, plan, critique, facts))
    print(f"  [report] Markdown -> {md_path}")

    # PDF (if fpdf2 installed)
    pdf_path = None
    if _FPDF_OK:
        pdf = PlanPDF()
        pdf._doc_title = title
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.title_page(title, city, date_str)

        pdf.add_page()
        pdf.h1("Executive Summary")
        pdf.body(narrative.get("executive_summary", ""))
        pdf.h2("Approach")
        pdf.body(narrative.get("approach_note", ""))

        _section_data_foundation(pdf, facts)
        _section_analysis(pdf, analysis)
        _section_solutions(pdf, plan)
        _section_review(pdf, critique)

        pdf.h1("Conclusion")
        pdf.body(narrative.get("conclusion", ""))

        pdf_path = f"{out_basename}.pdf"
        pdf.output(pdf_path)
        print(f"  [report] PDF -> {pdf_path}")
    else:
        print("  [report] PDF skipped (pip install fpdf2 to enable)")

    return {"narrative": narrative, "markdown_path": md_path, "pdf_path": pdf_path}


# Run Agent 5 standalone on whatever cached files exist:
#   python -m agents.agent_fifth_reporting
if __name__ == "__main__":
    import sys
    city = sys.argv[1] if len(sys.argv) > 1 else "Jaipur"

    def _load(filename):
        path = config.json_path(config.with_city(filename, city))
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"  [report] note: {path} not found")
        return None

    facts_in = _load(config.AGENT1_FACTS_CACHE)
    analysis_in = _load(config.AGENT2_ANALYSIS_CACHE)
    plan_in = _load("agent3_output.json")
    critique_in = _load("agent4_output.json")

    base = config.report_path(f"transport_plan_{config.city_slug(city)}")
    result = run_report_agent(plan_in, critique_in, analysis_in, facts_in, out_basename=base)
    print(f"[OK] Report: {result['markdown_path']}"
          + (f" + {result['pdf_path']}" if result["pdf_path"] else " (Markdown only)"))
