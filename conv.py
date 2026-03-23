import json
import random
import requests
import time
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Fallback so the script still works when only pypdf is installed.
if PyPDF2 is None:
    try:
        import pypdf as PyPDF2
    except ImportError:
        PyPDF2 = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
FALLBACK_MODELS = ["tinyllama:latest"]
TOTAL_SAMPLES = 200
BATCH_SIZE = 32
MAX_WORKERS = 4
REQUEST_TIMEOUT = 120
USE_LLM_FOR_PDFS = False

PDF_INPUT_DIR = Path("pdfs")
RAW_OUTPUT_DIR = Path("outputs/raw")
ALPACA_OUTPUT_DIR = Path("outputs/alpaca")
OUTPUT_FILE = ALPACA_OUTPUT_DIR / "indian_law_alpaca_200.json"

random.seed(42)

topics = [
    "Indian Penal Code (IPC)",
    "Constitution of India",
    "Indian Contract Act, 1872",
    "Law of Torts",
    "Criminal Procedure Code (CrPC)",
    "Civil Procedure Code (CPC)",
    "Information Technology Act, 2000",
    "Indian Evidence Act, 1872"
]

case_scenarios = [
    "A kills B during a sudden fight without premeditation.",
    "A refuses to honor a signed contract causing loss to B.",
    "A person is arrested without being informed of grounds.",
    "Government restricts free speech citing public order.",
    "A company leaks sensitive personal data of users.",
    "A driver negligently hits a pedestrian causing injury.",
    "Police obtain confession under coercion.",
    "A minor enters into a contract and later refuses performance."
]

instruction_types = [
    "Analyze the following case under {}.",
    "Provide a detailed legal answer under {}.",
    "Apply {} to the given scenario.",
    "Frame legal issues and decide the case under {}.",
    "Present moot court arguments for both sides under {}."
]

PDF_QUESTION_SETS = [
    {
        "label": "Case Summary",
        "instruction": "Summarize the judgment with facts, issues, law, analysis, and conclusion.",
        "input_focus": "Prepare a concise case summary for quick revision.",
        "output_focus": "Emphasize material facts, holding, and final outcome.",
    },
    {
        "label": "Issue Spotting",
        "instruction": "Identify and analyze the key legal issues decided in this case.",
        "input_focus": "Focus on precise legal questions framed by the court.",
        "output_focus": "Clearly separate primary and secondary issues.",
    },
    {
        "label": "Reasoning Analysis",
        "instruction": "Explain the court's reasoning and legal principles applied.",
        "input_focus": "Track how facts were connected to legal standards.",
        "output_focus": "Highlight reasoning steps and doctrinal tests.",
    },
    {
        "label": "Bilateral Arguments",
        "instruction": "Present arguments for both sides based on this case record.",
        "input_focus": "Structure submissions for petitioner and respondent.",
        "output_focus": "Balance both sides before reaching a conclusion.",
    },
    {
        "label": "Procedure Impact",
        "instruction": "Extract procedural history and explain its impact on final outcome.",
        "input_focus": "Focus on procedural steps and litigation path.",
        "output_focus": "Explain how procedure influenced substantive result.",
    },
    {
        "label": "Statute and Doctrine",
        "instruction": "List relevant statutes or doctrines and explain their application.",
        "input_focus": "Identify statutory references and legal doctrines.",
        "output_focus": "Tie each provision or doctrine to case facts.",
    },
    {
        "label": "Exam Answer",
        "instruction": "Draft an exam-style answer for this case for law students.",
        "input_focus": "Write in structured exam format with legal clarity.",
        "output_focus": "Prioritize issue-rule-application-conclusion flow.",
    },
    {
        "label": "Lawyer Brief",
        "instruction": "Write a practical lawyer briefing note based on this judgment.",
        "input_focus": "Focus on practitioner-oriented implications.",
        "output_focus": "Include actionable legal takeaways.",
    },
    {
        "label": "Precedent Value",
        "instruction": "Analyze precedent value and likely future legal impact.",
        "input_focus": "Assess binding value and citation potential.",
        "output_focus": "Discuss downstream interpretation risks.",
    },
    {
        "label": "Moot Preparation",
        "instruction": "Create a concise moot-court style analysis of this judgment.",
        "input_focus": "Prepare courtroom-ready points and rebuttals.",
        "output_focus": "Keep arguments compact and persuasive.",
    },
]


def mutate_scenario(base):
    variations = [
        base,
        base + " The incident occurred at night.",
        base + " There were multiple witnesses.",
        base + " The accused claims self-defense.",
        base + " The victim survived with injuries."
    ]
    return random.choice(variations)


def query_llm(prompt, retries=2):
    model_candidates = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]

    for model_name in model_candidates:
        for attempt in range(retries):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": -1,
                        "options": {
                            "temperature": 0.4,
                            "num_ctx": 4096,
                            "num_predict": 1000
                        }
                    },
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
                result = data.get("response", "").strip()
                if result:
                    if model_name != MODEL_NAME:
                        logger.info(f"Using fallback model: {model_name}")
                    return result
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request error for model {model_name} (attempt {attempt + 1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    time.sleep(0.8)
            except ValueError as e:
                logger.warning(
                    f"JSON parse error for model {model_name} (attempt {attempt + 1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    time.sleep(0.8)

    logger.error(f"Failed to get response after {retries} retries")
    return "ERROR"


def build_prompt(instruction, scenario, topic):
    return f"""
You are an Indian lawyer. Use real laws and case precedents.

Instruction:
{instruction}

Scenario:
{scenario}

Format:
Facts:
Issues:
Relevant Law:
Analysis:
Conclusion:
"""


def build_pdf_prompt(case_name, question_set, pdf_context):
    return f"""
You are an Indian lawyer. Use only the supplied case content.

Case:
{case_name}

Task:
{question_set['instruction']}

Task Focus:
{question_set['input_focus']}

Output Focus:
{question_set['output_focus']}

Case Content:
{pdf_context}

Required format:
Facts:
Issues:
Relevant Law:
Analysis:
Conclusion:
"""


def build_case_segments(pdf_text, segment_count=10, max_chars=1500):
    compact = " ".join((pdf_text or "").split())
    if not compact:
        return [""] * segment_count

    if len(compact) <= max_chars:
        return [compact for _ in range(segment_count)]

    segments = []
    max_start = len(compact) - max_chars
    step = max(1, max_start // max(1, segment_count - 1))

    for idx in range(segment_count):
        start = min(idx * step, max_start)
        end = start + max_chars
        segments.append(compact[start:end])

    return segments


def extract_case_signals(text):
    compact = " ".join((text or "").split())
    if not compact:
        return {
            "facts": ["Facts could not be extracted from the provided excerpt."],
            "issues": ["Core legal issue requires fuller record for precision."],
            "laws": ["Applicable law to be identified from complete judgment text."],
            "holding": "Outcome appears to turn on judicial interpretation of statutory/legal framework.",
        }

    sentence_candidates = re.split(r"(?<=[.!?])\s+", compact)
    sentence_candidates = [s.strip() for s in sentence_candidates if len(s.strip()) > 35]

    # Prefer readable narrative lines over citation-heavy fragments.
    def is_noisy(line):
        low = line.lower()
        noisy_tokens = [
            "equivalent citations", "air", "scr", "scc", "indiankanoon", "http", "bench:",
            "citation:", "author:", "petitioner:", "respondent:",
        ]
        digit_ratio = sum(ch.isdigit() for ch in line) / max(1, len(line))
        return digit_ratio > 0.18 or any(tok in low for tok in noisy_tokens)

    sentence_candidates = [s for s in sentence_candidates if not is_noisy(s)]
    if not sentence_candidates:
        sentence_candidates = [compact[:350]]

    fact_candidates = sentence_candidates[:6] if sentence_candidates else [compact[:300]]

    issue_keywords = ["whether", "issue", "question", "challenge", "valid", "invalid", "held", "dispute"]
    issues = [s for s in sentence_candidates if any(k in s.lower() for k in issue_keywords)]
    if not issues:
        issues = ["Whether the legal interpretation and application in this case was consistent with governing law."]

    act_matches = re.findall(r"\b([A-Z][A-Za-z\s().,&-]+? Act,?\s*\d{4})\b", compact)
    sec_matches = re.findall(
        r"\bSection\s+\d+[A-Za-z-]*(?:\s*\([^)]+\))?\b",
        compact,
        flags=re.IGNORECASE,
    )
    art_matches = re.findall(
        r"\bArticle\s+\d+[A-Za-z-]*(?:\s*\([^)]+\))?\b",
        compact,
        flags=re.IGNORECASE,
    )

    def normalize_law_ref(item):
        norm = re.sub(r"\s+", " ", item).strip(" .;,")
        norm = re.sub(r"\bsec\.?\b", "Section", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\bart\.?\b", "Article", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\bsection\b", "Section", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\barticle\b", "Article", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\bact\b", "Act", norm, flags=re.IGNORECASE)
        return norm

    laws = []
    for item in act_matches[:6] + sec_matches[:8] + art_matches[:6]:
        normalized = normalize_law_ref(item)
        if normalized and normalized not in laws:
            laws.append(normalized)

    # Keep highest-signal references first: Act names, then Articles, then Sections.
    def law_rank(item):
        low = item.lower()
        if " act" in low:
            return (0, low)
        if low.startswith("article"):
            return (1, low)
        if low.startswith("section"):
            return (2, low)
        return (3, low)

    laws = sorted(laws, key=law_rank)
    if not laws:
        low = compact.lower()
        inferred = []
        keyword_law_map = [
            (["income tax", "assessee", "assessment", "exemption"], "Income Tax Act, 1961"),
            (["contract", "tender", "bid", "auction", "consideration"], "Indian Contract Act, 1872"),
            (["criminal", "accused", "prosecution", "offence"], "Indian Penal Code"),
            (["evidence", "witness", "cross-examination", "admissible"], "Indian Evidence Act, 1872"),
            (["constitutional", "fundamental rights", "article"], "Constitution of India"),
            (["service", "pension", "appointment", "retirement"], "Service jurisprudence and applicable service rules"),
            (["procedure", "appeal", "revision", "writ"], "Procedural law principles under civil/criminal process"),
            (["arbitration", "award", "arbitral"], "Arbitration and Conciliation Act, 1996"),
        ]
        for keywords, law_name in keyword_law_map:
            if any(k in low for k in keywords):
                inferred.append(law_name)

        if inferred:
            laws = inferred[:4]
        else:
            laws = ["Relevant statutory framework and binding precedent referenced in the judgment excerpt."]

    holding = sentence_candidates[-1] if sentence_candidates else "Final holding requires full-text review."

    # De-duplicate near-identical lines.
    def dedupe_lines(lines, limit):
        seen = set()
        output = []
        for line in lines:
            key = re.sub(r"\W+", "", line.lower())[:140]
            if key and key not in seen:
                seen.add(key)
                output.append(line)
            if len(output) >= limit:
                break
        return output

    fact_candidates = dedupe_lines(fact_candidates, 4)
    issues = dedupe_lines(issues, 3)
    laws = dedupe_lines(laws, 6)

    return {
        "facts": fact_candidates[:3],
        "issues": issues[:3],
        "laws": laws[:5],
        "holding": holding,
    }


def build_task_specific_analysis(question_set, signals):
    label = question_set["label"]
    issues_text = "; ".join(signals["issues"][:2])
    laws_text = ", ".join(signals["laws"][:3])

    analysis_by_label = {
        "Case Summary": f"The case turns on {issues_text}. The court's approach links the factual matrix with {laws_text} and resolves the dispute through a structured application of legal standards.",
        "Issue Spotting": f"Primary issues include: {issues_text}. Secondary questions arise from statutory interpretation under {laws_text}.",
        "Reasoning Analysis": f"The reasoning proceeds by identifying controlling facts, applying {laws_text}, and testing whether the contested interpretation withstands doctrinal scrutiny.",
        "Bilateral Arguments": "For the petitioner, the strongest line is strict statutory reading and precedent consistency. For the respondent, equity, purposive interpretation, and factual context provide rebuttal strength.",
        "Procedure Impact": "Procedural developments shaped the permissible relief and narrowed adjudication scope, influencing how substantive rights were ultimately determined.",
        "Statute and Doctrine": f"Key legal anchors are {laws_text}. Their application indicates that doctrinal consistency, rather than ad hoc reasoning, drove the outcome.",
        "Exam Answer": f"Using IRAC structure: Issue ({issues_text}), Rule ({laws_text}), Application to extracted facts, and a reasoned conclusion consistent with the holding.",
        "Lawyer Brief": "Practically, counsel should focus on threshold maintainability, statutory preconditions, and precedent-backed framing when advising similarly situated clients.",
        "Precedent Value": "The decision has persuasive or binding value where similar statutory language and fact patterns recur; deviation risk rises when procedural posture differs materially.",
        "Moot Preparation": "Oral strategy should open with jurisdiction and rule framing, then pivot to factual fit, precedent hierarchy, and remedy calibration under bench questions.",
    }

    return analysis_by_label.get(
        label,
        f"The excerpt supports analysis focused on {question_set['output_focus'].lower()} through structured application of identified issues and governing law."
    )


def build_task_specific_sections(question_set, signals):
    label = question_set["label"]
    facts = signals["facts"]
    issues = signals["issues"]
    laws = signals["laws"]

    if label == "Case Summary":
        facts_block = [facts[0], facts[1] if len(facts) > 1 else facts[0], f"Outcome snapshot: {signals['holding']}"]
        issues_block = [issues[0], "What legal test did the court apply to resolve the dispute?"]
        laws_block = laws[:3]
    elif label == "Issue Spotting":
        facts_block = ["Material background extracted from the case excerpt:"] + facts[:2]
        issues_block = [f"Primary issue: {issues[0]}"] + [f"Secondary issue: {x}" for x in issues[1:3]]
        laws_block = ["Issue framing linked with:"] + laws[:3]
    elif label == "Reasoning Analysis":
        facts_block = ["Reasoning-relevant facts:"] + facts[:3]
        issues_block = ["Judicial reasoning questions:"] + issues[:2]
        laws_block = ["Doctrinal anchors in reasoning:"] + laws[:3]
    elif label == "Bilateral Arguments":
        facts_block = ["Common factual platform for both sides:"] + facts[:2]
        issues_block = [
            f"Petitioner-side controversy: {issues[0]}",
            f"Respondent-side controversy: {issues[1] if len(issues) > 1 else issues[0]}",
        ]
        laws_block = ["Authorities both sides would rely on:"] + laws[:3]
    elif label == "Procedure Impact":
        facts_block = ["Procedural checkpoints visible in excerpt:"] + facts[:2]
        issues_block = ["How procedure narrowed issues for adjudication.", issues[0]]
        laws_block = ["Procedure-linked legal basis:"] + laws[:3]
    elif label == "Statute and Doctrine":
        facts_block = ["Fact context for statutory application:"] + facts[:2]
        issues_block = ["Interpretive issue around statutory text.", issues[0]]
        laws_block = ["Statutes/doctrines identified:"] + laws[:4]
    elif label == "Exam Answer":
        facts_block = ["IRAC - Facts:"] + facts[:2]
        issues_block = ["IRAC - Issues:"] + issues[:2]
        laws_block = ["IRAC - Rules:"] + laws[:3]
    elif label == "Lawyer Brief":
        facts_block = ["Client-relevant background:"] + facts[:2]
        issues_block = ["Advisory risk questions:"] + issues[:2]
        laws_block = ["Actionable legal references:"] + laws[:3]
    elif label == "Precedent Value":
        facts_block = ["Precedent context:"] + facts[:2]
        issues_block = ["Ratio-sensitive issues for later citation:"] + issues[:2]
        laws_block = ["Likely citation anchors:"] + laws[:3]
    else:  # Moot Preparation
        facts_block = ["Moot proposition facts:"] + facts[:2]
        issues_block = ["Questions likely from the bench:"] + issues[:2]
        laws_block = ["Authorities for oral submissions:"] + laws[:3]

    return facts_block, issues_block, laws_block


def build_fallback_output(question_set, pdf_text, pdf_name, question_context):
    signals = extract_case_signals(question_context)
    facts_lines, issues_lines, laws_lines = build_task_specific_sections(question_set, signals)
    facts_block = "\n".join(f"- {item}" for item in facts_lines)
    issues_block = "\n".join(f"- {item}" for item in issues_lines)
    laws_block = "\n".join(f"- {item}" for item in laws_lines)
    analysis_text = build_task_specific_analysis(question_set, signals)
    conclusion_text = (
        f"For {question_set['label']}, a legally coherent answer prioritizes this endpoint: "
        f"{signals['holding'][:210]}"
    )

    return (
        "Facts:\n"
        f"{facts_block}\n\n"
        "Issues:\n"
        f"{issues_block}\n\n"
        "Relevant Law:\n"
        f"{laws_block}\n\n"
        "Analysis:\n"
        f"{analysis_text}\n\n"
        "Conclusion:\n"
        f"{conclusion_text}"
    )


def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    if not PyPDF2:
        logger.warning("PyPDF2/pypdf not installed. Install with: pip install pypdf")
        return None

    try:
        text = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return '\n'.join(text)
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return None


def get_pdf_files(directory=PDF_INPUT_DIR):
    """Get all PDF files in the pdfs directory."""
    files = list(Path(directory).glob("*.pdf")) + list(Path(directory).glob("*.PDF"))
    # Windows paths are case-insensitive; dedupe by normalized lowercase path.
    unique = {}
    for file_path in files:
        unique[str(file_path).lower()] = file_path
    return sorted(unique.values(), key=lambda p: p.name.lower())


def is_valid(entry):
    out = entry["output"]
    required = ["Facts:", "Issues:", "Relevant Law:", "Analysis:", "Conclusion:"]
    return out and not out.startswith("ERROR") and all(r in out for r in required)


def hash_entry(e):
    return hashlib.md5((e["instruction"] + e["input"] + e["output"]).encode()).hexdigest()


def generate_entry(instruction=None, scenario=None, topic=None):
    """Generate a single entry. Use provided values or generate from defaults."""
    if not all([instruction, scenario, topic]):
        topic = random.choice(topics)
        scenario = mutate_scenario(random.choice(case_scenarios))
        instruction = random.choice(instruction_types).format(topic)

    prompt = build_prompt(instruction, scenario, topic)
    output = query_llm(prompt)

    return {
        "instruction": instruction.strip(),
        "input": scenario.strip(),
        "output": output.strip()
    }


def generate_dataset_from_scenarios():
    """Generate dataset from predefined scenarios."""
    dataset, seen = [], set()

    while len(dataset) < TOTAL_SAMPLES:
        batch_size = min(BATCH_SIZE, TOTAL_SAMPLES - len(dataset))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(generate_entry) for _ in range(batch_size)]

            for f in as_completed(futures):
                try:
                    e = f.result()

                    if not is_valid(e):
                        continue

                    h = hash_entry(e)
                    if h in seen:
                        continue

                    seen.add(h)
                    dataset.append(e)

                    if len(dataset) >= TOTAL_SAMPLES:
                        break
                except Exception as e:
                    logger.error(f"Error generating entry: {e}")
                    continue

        logger.info(f"Progress: {len(dataset)}/{TOTAL_SAMPLES}")

    ALPACA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    logger.info("Dataset generation from scenarios complete")
    return OUTPUT_FILE


def extract_key_information(pdf_text, pdf_name):
    """Extract key legal information from PDF text."""
    lines = pdf_text.split('\n')

    # Extract parties (usually at the beginning)
    parties = "Unknown Parties"
    for line in lines[:50]:
        if 'vs' in line.lower() or 'v.' in line.lower():
            parties = line.strip()
            break

    # Extract case number/citation
    case_number = "Not specified"
    for line in lines[:30]:
        if any(keyword in line.lower() for keyword in ['reported in', 'citation', 'case no', '(', ')']):
            if len(line.strip()) < 100:
                case_number = line.strip()
                break

    # Extract judgment date
    judgment_date = "Not specified"
    for line in lines[:50]:
        date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', line)
        if date_match:
            judgment_date = date_match.group()
            break

    # Extract key facts (first substantial paragraphs)
    facts = []
    for line in lines[:100]:
        if line.strip() and len(line.strip()) > 20 and not any(x in line.lower() for x in ['page', 'court', '===', '---']):
            facts.append(line.strip())
            if len(facts) >= 3:
                break
    facts_text = ' '.join(facts)[:500] if facts else "Case facts extracted from document"

    # Extract issues/judgement (usually marked or in structured format)
    issues = "Legal issues and judgment extracted from the court document"
    for line in lines:
        if any(keyword in line.lower() for keyword in ['issue', 'held', 'decided', 'judgment']):
            issues = line.strip()[:200]
            if len(issues) > 50:
                break

    # Create structured output
    output = f"""Facts:
{parties}
Case: {case_number}
Judgment Date: {judgment_date}
{facts_text}

Issues:
{issues}

Relevant Law:
Indian law principles and legal framework applicable to this case as per Indian Constitution, IPC, CPC, and other relevant statutes.

Analysis:
The court analyzed the case based on established legal precedents and applicable Indian law provisions.

Conclusion:
The judgment provides resolution based on legal analysis of the facts and applicable law."""

    return output


def process_pdf_files(pdf_dir=PDF_INPUT_DIR):
    """Process all PDF files in pdfs and convert outputs to JSON files."""
    pdf_files = get_pdf_files(pdf_dir)

    if not pdf_files:
        logger.warning(f"No PDF files found in: {pdf_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")

    if not PyPDF2:
        logger.error("PyPDF2/pypdf is required to process PDFs. Install with: pip install pypdf")
        return []

    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ALPACA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_files = []

    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")

        # Extract text from PDF
        pdf_text = extract_pdf_text(pdf_path)
        if not pdf_text:
            logger.warning(f"Could not extract text from {pdf_path.name}")
            continue

        try:
            # Save raw extraction for this PDF
            raw_output_file = RAW_OUTPUT_DIR / f"{pdf_path.stem}_raw.json"
            with open(raw_output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_file": pdf_path.name,
                        "text": pdf_text,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            # Create instruction and topic from PDF filename
            pdf_name = pdf_path.stem
            case_title = pdf_name.replace('_', ' ')
            case_segments = build_case_segments(pdf_text, segment_count=len(PDF_QUESTION_SETS), max_chars=1500)

            # Create 10 alpaca entries per case with different instruction types.
            dataset = []
            for idx, question_set in enumerate(PDF_QUESTION_SETS, start=1):
                question_context = case_segments[idx - 1]
                instruction = f"{question_set['instruction']} (Case: {case_title})"

                # Fast and stable default mode: avoid 10 model calls per PDF.
                if USE_LLM_FOR_PDFS:
                    model_prompt = build_pdf_prompt(case_title, question_set, question_context)
                    output_text = query_llm(model_prompt)
                    if output_text == "ERROR" or not output_text.strip():
                        output_text = build_fallback_output(question_set, pdf_text, pdf_name, question_context)
                else:
                    output_text = build_fallback_output(question_set, pdf_text, pdf_name, question_context)

                entry = {
                    "instruction": instruction,
                    "input": (
                        f"Case: {case_title}\n"
                        f"Task Type: {question_set['label']}\n"
                        f"Task Focus: {question_set['input_focus']}\n"
                        f"Output Focus: {question_set['output_focus']}\n\n"
                        f"Case Excerpt:\n{question_context}"
                    ),
                    "output": output_text,
                }

                if not is_valid(entry):
                    entry["output"] = build_fallback_output(question_set, pdf_text, pdf_name, question_context)

                dataset.append(entry)

            # Save alpaca json file with same name as PDF
            alpaca_file = ALPACA_OUTPUT_DIR / f"{pdf_path.stem}.json"
            with open(alpaca_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)

            logger.info(f"Created raw: {raw_output_file.name}")
            logger.info(f"Created alpaca: {alpaca_file.name}")
            output_files.append(str(alpaca_file))

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            continue

    if output_files:
        logger.info(f"Successfully converted {len(output_files)} PDF(s) to Alpaca JSON")

    return output_files


def generate_dataset():
    """Main function to generate/process datasets."""
    pdf_files = get_pdf_files(PDF_INPUT_DIR)

    if pdf_files:
        logger.info(f"PDF files detected in {PDF_INPUT_DIR}. Processing PDFs and extracting key information...")
        try:
            output_files = process_pdf_files(PDF_INPUT_DIR)
            if output_files:
                logger.info(f"Successfully processed {len(output_files)} file(s)")
                logger.info(f"Raw outputs: {RAW_OUTPUT_DIR}")
                logger.info(f"Alpaca outputs: {ALPACA_OUTPUT_DIR}")
            else:
                logger.warning("No files were successfully processed")
        except Exception as e:
            logger.error(f"Error during PDF processing: {e}")
    else:
        logger.info("No PDF files found. Generating dataset from scenarios...")
        try:
            output_file = generate_dataset_from_scenarios()
            logger.info(f"Generated file: {output_file}")
        except Exception as e:
            logger.error(f"Error during scenario-based generation: {e}")


if __name__ == "__main__":
    generate_dataset()
