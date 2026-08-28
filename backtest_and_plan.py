name: OBV Strategy Backtest

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run-backtest:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas

      - name: Run Strategy Backtest
        run: |
          python backtest_and_plan.py

      - name: Commit & Push Results
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add data/backtest_report.json
          if git diff --staged --quiet; then
            echo "No changes."
          else
            git commit -m "Update OBV Backtest Report [$(date -u +'%Y-%m-%d %H:%M UTC')]"
            git push
          fi
