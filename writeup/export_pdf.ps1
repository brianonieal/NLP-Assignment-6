# Exports writeup\Onieal_HW6.docx to PDF with Microsoft Word, so equations render as Word draws them.
param([string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $true)
    $doc.ExportAsFixedFormat($Pdf, 17)
    $doc.Close($false)
    "exported $Pdf"
} finally {
    $word.Quit()
}
