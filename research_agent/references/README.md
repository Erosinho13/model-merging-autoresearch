# Convert reference papers to Markdown

1. Download a paper as a PDF and place it in `research_agent/references/`.
2. Create and activate a separate environment, then install Marker:

   ```bash
   conda create -n marker python=3.11
   conda activate marker
   pip install marker-pdf
   ```

3. On macOS, install llama.cpp:

   ```bash
   brew install llama.cpp
   ```

4. From the repository root, run the conversion script yourself:

   ```bash
   bash research_agent/references/convert_paper.sh research_agent/references/ties.pdf
   ```

The script creates a folder named after the PDF, moves the PDF inside, and runs
`marker_single` with an explicit output directory.