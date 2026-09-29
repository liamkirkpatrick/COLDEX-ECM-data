# AGENTS.md

## Project

See `README.md` for the project overview, repository structure, and setup instructions.

## Environment

- Use the environment defined in `environment.yml`. Locally, this conda environment is called 'allan_hills_exp'.
- You can install new packages to this environment, but ask permission first.

## Repository Structure

See `README.md`.

## Data and Scientific Conventions

## Plotting

- Whenever making plots, use full latex forms of values ('\delta^{18}O' not d18O). After an axis label, report units within paraenthesis.
- Make depth increase down, age increase to the right. The default should always be to put age on the x-axis and depth on the y-axis unless I specify otherwise.
- Whenever you make a plot, save it to figures/[appropriate_subfolder]/[figure_name].png. 

## Agent-Specific Instructions

- Do not commit files under `reference/`.
- Only commit changes under `data/` if explicitly instructed to. Otherwise, keep working files within agentic_workspace.
- Use the environment defined in `environment.yml`.
- Keep exploratory work in `agentic_workspace/`. This should include both intermediate datafiles and code. Make appropritate sub-folders for discrete workjstreams.
- If exploratory work moves into something worth of the main workstream, we'll move it into `scripts/`
- My default is jupyter notebooks for my final code, but you can work in .py files for exploratory work.

## Units

**Temperature** - default to degrees Celcius
**d18O, dD, dxs, dln** - default to parts per thousand (per mille)
**Depth** - default to m