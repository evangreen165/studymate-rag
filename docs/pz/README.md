# Пояснительная записка

`ПЗ_Ассистент_по_учебным_материалам_Кузьма.docx` — пояснительная записка по СТП 01-2024.
`ПЗ_Ассистент_по_учебным_материалам_Кузьма.pdf` — версия для просмотра и печати.

Сборка:

```bash
uv run python docs/pz/build_pz.py
```

Затем в Word обновите поля и оглавление (Ctrl+A, F9) и экспортируйте PDF. В Windows это можно
сделать автоматически:

```powershell
$w = New-Object -ComObject Word.Application
$d = $w.Documents.Open("docs/pz/ПЗ_Ассистент_по_учебным_материалам_Кузьма.docx")
$d.Fields.Update(); $d.TablesOfContents.Item(1).Update()
$d.Save(); $d.ExportAsFixedFormat("docs/pz/ПЗ_Ассистент_по_учебным_материалам_Кузьма.pdf", 17)
$d.Close(); $w.Quit()
```

Исходный текст разделов — в `content/*.md`. Числа в тексте подставляются из `data/manifest.json`,
`data/index_meta.json` и `results/ragas_test.json`, поэтому после пересчёта метрик достаточно
пересобрать документ.
