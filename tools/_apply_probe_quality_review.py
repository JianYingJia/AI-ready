"""One-time editorial update for the independently sourced CLL probe catalogue."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "inputs/conditional_value/cll_independent_capability_probes_v1.jsonl"


OPTION_UPDATES: dict[str, list[str]] = {
    # Diagnosis and differential: use nearby hematologic alternatives and close thresholds.
    "cll-diag-01": [
        "At least 5 × 10^9/L clonal B cells",
        "At least 0.5 × 10^9/L clonal B cells",
        "At least 10 × 10^9/L clonal B cells",
        "At least 5 × 10^9/L total lymphocytes without proof of clonality",
    ],
    "cll-diag-02": [
        "Overt CLL",
        "Monoclonal B-cell lymphocytosis (MBL)",
        "Small lymphocytic lymphoma (SLL)",
        "Richter transformation",
    ],
    "cll-diag-03": [
        "CD5−, CD19+, bright CD20+, CD23− clonal B cells",
        "CD5+, CD19+, bright CD20+, CD23− clonal B cells",
        "CD5+, CD19+, dim CD20+, CD23+ clonal B cells",
        "CD5−, CD19+, dim CD20+, CD23− clonal B cells",
    ],
    "cll-diag-06": [
        "Fluorescence in situ hybridization",
        "Flow cytometry",
        "IGH variant analysis",
        "Bone marrow aspiration and biopsy",
    ],
    "cll-diag-07": [
        "Coombs-positive hemolytic anemia",
        "Depressed immunoglobulin levels",
        "Hyperviscosity caused by elevated IgM",
        "Progressive clonal B-cell lymphocytosis",
    ],
    "cll-diag-08": [
        "CD5, CD19, and CD23",
        "CD2, CD3, and CD8",
        "CD19, CD20, and CD23",
        "CD2, CD16, and CD56",
    ],
    "cll-diag-09": [
        "Neutropenia and a history of rheumatoid arthritis",
        "Hyperviscosity and an elevated IgM level",
        "Coombs-positive hemolysis and depressed immunoglobulin levels",
        "A clonal CD5+/CD23+ B-cell population and lymphadenopathy",
    ],
    "cll-diag-05": [
        "CT is usually not required",
        "CT should be performed routinely at diagnosis",
        "PET-CT is preferred to CT for baseline staging",
        "CT should be deferred until after bone marrow biopsy",
    ],
    "cll-diag-10": [
        "Peripheral blood and spleen",
        "Blood, bone marrow, and lymphatic tissues",
        "Bone marrow, spleen, and liver",
        "Lymph nodes, spleen, and liver",
    ],
    # Pathology and immunophenotype: alternatives differ by one biologically relevant feature.
    "cll-path-01": [
        "Morphologically mature and immunologically mature",
        "Morphologically immature and immunologically less mature",
        "Morphologically mature but immunologically less mature",
        "Morphologically immature but immunologically mature",
    ],
    "cll-path-02": [
        "B-cell antigens CD19/CD20 together with CD10",
        "B-cell antigens CD19/CD20 together with CD3",
        "T-cell antigens CD2/CD3 together with CD8",
        "B-cell antigens CD19/CD20 together with T-cell antigen CD5",
    ],
    "cll-path-03": [
        "Relatively low expression",
        "Usually absent expression",
        "Expression comparable to normal peripheral-blood B cells",
        "Relatively high expression",
    ],
    "cll-path-04": [
        "Coexpression of kappa and lambda on the same clonal population",
        "A single light chain, either kappa or lambda",
        "Complete absence of both light chains",
        "A polytypic mixture of kappa- and lambda-expressing B cells",
    ],
    "cll-path-05": [
        "Absent",
        "Bright",
        "Dim",
        "Equivalent to normal peripheral-blood B cells",
    ],
    "cll-path-06": [
        "CD10 and CD19",
        "CD19 and CD20",
        "CD20 and CD22",
        "CD5 and CD23",
    ],
    "cll-path-08": [
        "FISH identifies all TP53 sequence variants even when 17p is intact",
        "Some patients have a TP53 mutation without del(17p) on FISH",
        "A TP53 mutation is not clinically relevant when FISH shows intact 17p",
        "TP53 sequencing is used to confirm trisomy 12",
    ],
    # Complications: replace unrelated diseases with competing CLL complications or close outcomes.
    "cll-comp-01": [
        "Hemorrhage and infection",
        "Autoimmune hemolysis and immune thrombocytopenia",
        "Tumor lysis syndrome and Richter transformation",
        "Atrial fibrillation and treatment-related bleeding",
    ],
    "cll-comp-02": [
        "Marrow-failure anemia, neutropenia, and thrombocytopenia",
        "Coombs-positive hemolytic anemia, immune thrombocytopenia, and depressed immunoglobulins",
        "Tumor lysis syndrome, infection, and hemorrhage",
        "Bleeding, atrial fibrillation, and hypertension",
    ],
    "cll-comp-03": [
        "Autoantibody production against erythrocytes",
        "Treatment-related neutropenia alone",
        "Hypogammaglobulinemia with impaired humoral defense",
        "An isolated increase in serum immunoglobulins",
    ],
    "cll-comp-04": [
        "Cytomegalovirus infection",
        "Hepatitis B infection",
        "Herpes simplex infection",
        "Herpes zoster",
    ],
    "cll-comp-05": [
        "Pneumocystis infection and Candida infection",
        "Herpes zoster and influenza",
        "Pneumococcal infection and Haemophilus infection",
        "Cytomegalovirus infection and Aspergillus infection",
    ],
    "cll-comp-06": [
        "Fewer bacterial infections together with improved survival",
        "Fewer bacterial infections, but no survival effect",
        "No reduction in bacterial infections, but improved survival",
        "A delayed first infection without a reduction in bacterial infections",
    ],
    "cll-comp-08": [
        "Corticosteroids",
        "High-dose immune globulin",
        "Rituximab",
        "Fludarabine",
    ],
    "cll-comp-10": [
        "Neutropenia and bacterial infection",
        "Bleeding and atrial fibrillation",
        "Tumor lysis syndrome and renal failure",
        "Autoimmune hemolysis and immune thrombocytopenia",
    ],
    "cll-comp-07": [
        "Rai stage 0 or I",
        "Rai stage II",
        "At any stage",
        "Rai stage III or IV",
    ],
    # Prognosis and transformation: all distractors are CLL-relevant factors or close intervals.
    "cll-prog-01": [
        "Absence of somatic hypermutation in the IGH variable region",
        "TP53 disruption accompanied by deletion of chromosome 17p",
        "A significant number of IGH variants (mutated IGH)",
        "A complex karyotype containing multiple chromosomal abnormalities",
    ],
    "cll-prog-03": [
        "del(17p)",
        "Isolated del(13q)",
        "Trisomy 12",
        "del(11q)",
    ],
    "cll-prog-04": [
        "When they result from autoimmune destruction",
        "When they result from extensive marrow involvement by CLL",
        "When they follow an acute infection",
        "When they are transient after treatment",
    ],
    "cll-prog-06": [
        "Doubling in 18 to 24 months",
        "A stable count over one year",
        "A decrease after an intercurrent infection resolves",
        "Doubling in less than one year",
    ],
    "cll-prog-08": [
        "Accelerated CLL without a new lymphoma",
        "Diffuse large B-cell lymphoma",
        "Hodgkin lymphoma",
        "Lymphoplasmacytic lymphoma",
    ],
    "cll-prog-09": [
        "Mutated IGH, isolated del(13q), and a noncomplex karyotype",
        "Unmutated IGH, wild-type TP53 and NOTCH1, and intact CDKN2A/B",
        "Unmutated IGH, TP53 or NOTCH1 variants, CDKN2A/B loss, and complex karyotype",
        "Mutated IGH, low beta-2-microglobulin, and stable lymphocyte counts",
    ],
    "cll-prog-10": [
        "Clinical stage, lymphocyte doubling time, sex, LDH, and del(13q)",
        "Age, performance status, hemoglobin, platelet count, and LDH",
        "IGH status, del(11q), lymphocyte count, sex, and splenomegaly",
        "IGH status, clinical stage, age, TP53 status, and serum beta-2-microglobulin",
    ],
    "cll-prog-05": [
        "Stable lymph-node size without systemic symptoms",
        "Low-count MBL without lymphadenopathy or systemic symptoms",
        "Rapidly growing lymph nodes with recurrent fever or soaking night sweats",
        "Long-standing small lymph nodes without systemic symptoms",
    ],
    # Treatment indication: use near-threshold values and realistic competing management choices.
    "cll-indic-01": [
        "Careful observation",
        "A BTK inhibitor as immediate therapy",
        "Fixed-duration venetoclax plus an anti-CD20 antibody",
        "Chemoimmunotherapy",
    ],
    "cll-indic-02": [
        "Start chemoimmunotherapy solely because of the marker",
        "Continue observation rather than treat solely for the marker",
        "Start targeted therapy solely because of the marker",
        "Refer directly for transplant solely because of the marker",
    ],
    "cll-indic-03": [
        "The platelet count by itself is sufficient to initiate treatment",
        "One month at this platelet level is sufficient to initiate treatment",
        "This alone does not automatically require treatment",
        "The finding becomes sufficient after six months of stability",
    ],
    "cll-indic-04": [
        "Hemoglobin below 12 g/dL or platelets below 100 × 10^9/L",
        "Hemoglobin below 11 g/dL or platelets below 75 × 10^9/L",
        "Hemoglobin below 8 g/dL or platelets below 20 × 10^9/L",
        "Hemoglobin below 10 g/dL or platelets below 50 × 10^9/L",
    ],
    "cll-indic-05": [
        "At least 6 cm below the left costal margin",
        "At least 4 cm below the left costal margin",
        "At least 8 cm below the left costal margin",
        "At least 10 cm below the left costal margin",
    ],
    "cll-indic-06": [
        "At least 5 cm in the longest diameter",
        "At least 10 cm in the longest diameter",
        "At least 15 cm in the longest diameter",
        "At least 20 cm in the longest diameter",
    ],
    "cll-indic-07": [
        "A rise of at least 25% over two months or a doubling time under six months",
        "A rise of at least 50% over six months or a doubling time under twelve months",
        "A rise of at least 50% over two months or doubling time under six months",
        "A rise of at least 25% over six months or a doubling time under twelve months",
    ],
    "cll-indic-10": [
        "Unintentional loss of at least 5% within six months",
        "Unintentional loss of at least 10% within six months",
        "Unintentional loss of at least 10% within twelve months",
        "Unintentional loss of at least 15% within twenty-four months",
    ],
    "cll-indic-09": [
        "Symptomatic or functionally important involvement of organs such as skin, kidney, lung, or spine",
        "An asymptomatic renal imaging finding without functional impairment",
        "A stable pulmonary finding unrelated to CLL and without symptoms",
        "Previously treated spinal disease with no current symptoms or functional impairment",
    ],
    # Treatment selection: alternatives use the same therapeutic vocabulary as the source.
    "cll-select-01": [
        "Acalabrutinib, zanubrutinib, and venetoclax",
        "Zanubrutinib, rituximab, and ibrutinib",
        "Acalabrutinib, zanubrutinib, and ibrutinib",
        "Ibrutinib, venetoclax, and obinutuzumab",
    ],
    "cll-select-02": [
        "Acalabrutinib or zanubrutinib",
        "Fludarabine or cyclophosphamide",
        "Prednisone or dexamethasone",
        "Obinutuzumab or rituximab",
    ],
    "cll-select-03": [
        "A chemotherapy-free approach",
        "Chlorambucil-based chemoimmunotherapy",
        "FCR chemoimmunotherapy",
        "Bendamustine-rituximab chemoimmunotherapy",
    ],
    "cll-select-04": [
        "For del(17p) or TP53-altered CLL without suspected transformation",
        "When Richter syndrome with histological progression is clinically suspected",
        "For symptomatic progressive CLL without suspected transformation",
        "For isolated del(13q) with asymptomatic disease",
    ],
    "cll-select-06": [
        "Switch to an anti-CD20 antibody alone",
        "Switch to chemoimmunotherapy",
        "Switch to a PI3K inhibitor",
        "Switch to a BTK inhibitor",
    ],
    "cll-select-07": [
        "Venetoclax-based therapy",
        "A different covalent BTK inhibitor",
        "Chemoimmunotherapy without considering prior therapy",
        "Observation despite documented progression",
    ],
    "cll-select-08": [
        "Continue the same covalent BTK inhibitor at an increased dose",
        "Switch to an alternative covalent BTK inhibitor",
        "Assume progression and switch to venetoclax in every case",
        "Stop CLL-directed therapy permanently",
    ],
    "cll-select-09": [
        "It irreversibly binds C481 more strongly than a covalent inhibitor",
        "It bypasses BTK and inhibits BCL2 instead",
        "It was developed not to rely on the C481 binding site",
        "It restores the wild-type BTK sequence",
    ],
    "cll-select-10": [
        "Phosphoinositide 3-kinase delta (PI3Kδ)",
        "B-cell lymphoma 2 protein (BCL2)",
        "Spleen tyrosine kinase (SYK)",
        "Bruton tyrosine kinase (BTK)",
    ],
    "cll-select-05": [
        "Standard-dose fludarabine, cyclophosphamide, and rituximab",
        "Myeloablative allogeneic stem-cell transplantation",
        "Newer biological agents such as ibrutinib or venetoclax",
        "Intensive rituximab, cyclophosphamide, doxorubicin, vincristine, and prednisone",
    ],
    "cll-select-08": [
        "Resume the same covalent BTK inhibitor at full dose",
        "Switch to an alternative covalent BTK inhibitor",
        "Switch directly to chemoimmunotherapy",
        "Treat the intolerance as class resistance and use R-CHOP",
    ],
}


def main() -> None:
    rows = [json.loads(line) for line in CATALOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for row in rows:
        probe_id = row["probe_id"]
        if probe_id not in OPTION_UPDATES:
            continue
        old_answer = row["answer_text"]
        answer_index = row["answer_index"]
        new_options = OPTION_UPDATES[probe_id]
        if len(new_options) != 4 or len(set(new_options)) != 4:
            raise ValueError(f"{probe_id}: updated options must contain four distinct values")
        if new_options[answer_index] != old_answer:
            raise ValueError(f"{probe_id}: answer position or answer text changed")
        row["options"] = new_options
        seen.add(probe_id)
    if seen != set(OPTION_UPDATES):
        raise ValueError(f"Unknown update IDs: {sorted(set(OPTION_UPDATES) - seen)}")
    with CATALOG.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Updated {len(seen)} of {len(rows)} probes")


if __name__ == "__main__":
    main()
