pandoc -o "TITLE.epub" `
    --epub-cover-image=cover.png `
    --metadata title="TITLE: AI Edit" `
    --metadata author="AUTHOR, LLM" `
    --metadata lang="en-US" `
    --css=epub.css `
    --toc `
    --toc-depth=2 `
    --split-level=1 `
    "INPUT FILE.md"