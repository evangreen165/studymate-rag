# Презентация защиты

`Презентация_защита_StudyMate.pptx` — 15 слайдов, 16:9. PDF-версия предназначена для просмотра
и резервной копии на защите.

Сборка:

```bash
uv run python docs/presentation/build_pptx.py
```

Для обновления PDF в Windows можно использовать PowerPoint COM:

```powershell
$p = New-Object -ComObject PowerPoint.Application
$d = $p.Presentations.Open("docs/presentation/Презентация_защита_StudyMate.pptx")
$d.SaveAs("docs/presentation/Презентация_защита_StudyMate.pdf", 32)
$d.Close(); $p.Quit()
```
