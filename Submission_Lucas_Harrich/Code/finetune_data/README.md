# Austrian Corporate Tax Seed Dataset

Lucas Harrich

Dieses Verzeichnis enthält ein österreichisches QA-Dataset für Supervised Fine-Tuning zur Körperschaftsteuer.

Dateien:
- `austrian_corp_tax_seed_sft.jsonl`: direkt für SFT/Chat-Training nutzbar
- `austrian_corp_tax_seed_qa.csv`: einfache QA-Tabelle mit Quellenangaben
- `build_austrian_corp_tax_seed.py`: Generator für beide Datensätze

Format:
- Jede JSONL-Zeile enthält `id`, `topic`, `source_title`, `source_url` und `messages`
- `messages` besteht aus `system`, `user` und `assistant`

Hinweise:
- Der Datensatz umfasst aktuell 146 Beispiele.
- Die Antworten sind bewusst kurz und stilistisch einheitlich gehalten.
- Die Quellen stammen aus offiziellen österreichischen Rechtsquellen, vor allem RIS und Findok/BMF.
- Für Evaluation solltest du weiterhin ein separates Holdout-Set zurückhalten.
- Falls du weitere Themenblöcke ergänzen willst, ändere die Faktensammlung im Generator und führe das Skript erneut aus.

Empfohlene nächste Schritte:
1. Antworten stichprobenartig manuell prüfen und sprachlich vereinheitlichen.
2. Ein kleines Holdout-Set abtrennen.
3. Das JSONL mit `datasets` laden.
4. Mit LoRA oder QLoRA auf einem kleinen Instruct-Modell trainieren.
