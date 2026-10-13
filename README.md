<div align="center">

⚡ Power Flow Case Comparison Engine

Compare power-system cases. Find differences. Focus your investigation.

A Python tool for comparing network models and power-flow results, surfacing discrepancies, and producing readable comparison reports.

Quick Start · Run the Project · Documentation · Author

</div>

✨ What it does

🔎 Compares cases to identify differences in network models and electrical results.

📊 Summarizes discrepancies with statistical insights and investigation priorities.

📝 Generates reports that help you review results and decide what to investigate next.

🧰 Offers multiple entry points for a quick example, your own cases, CLI usage, or focused demos.

🚀 Quick Start

1. Create a virtual environment

    - Run these commands from the project root.

          python -m venv .venv

    -- Windows (PowerShell)

          .venv\Scripts\Activate.ps1

Linux / macOS

          source .venv/bin/activate

2. Install dependencies

          python -m pip install --upgrade pip
          python -m pip install -r requirements.txt

Python version: Use a Python version supported by the project and its pinned dependencies. Add the tested version here before publishing.

🧭 Choose How to Run

1. Run the example

Run the project's example entry point:

          python Examples/example.py

2. Compare your own cases

          Open config/settings.yaml.

Configure the paths and formats for Case A and Case B.

Set the desired output location and other comparison options.

Confirm that the external power-flow files exist.

Then run from the project root:

          python -m src.Main

This mode requires external power-flow files and a valid config/settings.yaml.

3. Use the command-line interface

Use the CLI to run an example case:

          python -m Examples.cli run IEEE14bus

See the available options:

          python -m Examples.cli run --help

For example-listing or other commands, consult the CLI help and the project's current CLI implementation.

4. Explore individual demo scenarios

Examples/demo.py contains several scenarios that are commented out.

Open Examples/demo.py.

Uncomment the scenario you want to try.

Run the file from the project root:

          python Examples/demo.py

Run one scenario at a time. Check the selected scenario for any additional input-file requirements.

📂 Output

Generated reports and case files are written to the locations defined by the selected example or configuration. For your own-case run, check Output_path in config/settings.yaml.

📖 Documentation

The detailed help document will be added to the Docs/ folder.

Once available, it can cover configuration options, supported input formats, report interpretation, and troubleshooting.

👤 Author

Name: Kashyap Vasudevan

Email: [kashyap.vasudevan@gmail.com]


📄 License

License inherits the licenses of the packages used in the tool
