How to rerun climateNLP2025 scenario's:

replace review.py with /climateNLP2025/code/review.py
replace /query/querier.py with /climateNLP2025/code/querier.py
replace /prompts/prompt_templates.py with /climateNLP2025/code/prompt_templates.py

for excess nutrients, create folder /docs/GBF_T7_all_en
for pesticides and other hazardous chemicals, create folder /docs/GBF_T7_all_ps
for plastics, create folder /docs/GBF_T7_all_pl

copy all files from /climateNLP2025/data/GBF_T7_20250129 to these three folders

in folder /docs/GBF_T7_all_en, create subfolder review and copy files questions_en.txt and synthesize_prompt.txt in that subfolder
in folder /docs/GBF_T7_all_ps, create subfolder review and copy files questions_ps.txt and synthesize_prompt.txt in that subfolder
in folder /docs/GBF_T7_all_pl, create subfolder review and copy files questions_pl.txt and synthesize_prompt.txt in that subfolder

run review.py for folder /docs/GBF_T7_all_en, /docs/GBF_T7_all_ps and /docs/GBF_T7_all_pl

run /scripts/sum_clusters.py on resulting files /docs/GBF_T7_all_en/review/yyyy_mm_dd_hhhour_mmmin_sssec/synthesis.tsv, /docs/GBF_T7_all_ps/review/yyyy_mm_dd_hhhour_mmmin_sssec/synthesis.tsv and /docs/GBF_T7_all_pl/review/yyyy_mm_dd_hhhour_mmmin_sssec/synthesis.tsv