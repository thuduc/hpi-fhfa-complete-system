## GenAI Proof of Concept: generate an end-to-end system implementation of FHFA's Repeat-Sales Aggregation Index model whitepaper

The purpose of this proof of concept is to find out if an LLM can take an econometric model whitepaper and generate an end-to-end system implementation. The whitepaper used for this PoC is FHFA's Repeat-Sales Aggregation Index: https://www.fhfa.gov/document/wp2101.pdf (saved as [hpi_fhfa_whitepaper.pdf](hpi_fhfa_whitepaper.pdf))

### LLM & AI Tool
* LLM used: Claude Opus 4 (best coding LLM) - https://www.anthropic.com/claude/opus
* AI tool used: Claude Code (best coding CLI due to its integration with Clause 4 LLMs) - https://www.anthropic.com/claude-code

### Development Process: 
* Step 1 - request Opus 4 LLM to generate a Product Requirements Document (PRD) by analyzing the provided model whitepaper
* Step 2 - request Claude Code (using Opus 4 as the model) to generate a detailed system implementation plan by analyzing the PRD generated in Step 1. For our POC, we will request Claude Code to generate an implementation plans base on Python and Pandas, stored in [IMPLEMENTATION_PLAN_PANDAS.md](IMPLEMENTATION_PLAN_PANDAS.md)
* Step 3 - request Claude Code (with Opus 4) to implement all phases of this plan. Each plan includes requirements for comprehensive test coverage (both unit and integration tests)

### How we verified the accuracy of the Product Requirements Document (PRD)
We used Claude Opus 4 LLM to generate the PRD from the whitepaper. In order to verify the accuracy of the PRD, we used 2 additional LLMs:
* ChatGPT o3 response: "A line-by-line comparison of the PRD with the mathematical definitions and data-handling rules in Contat & Larson (2022) shows that every equation, variable name, coefficient definition and threshold appearing in the PRD is consistent with the white-paper. No corrections are required."
* Gemini 2.5 Pro response: "The Product Requirements Document (PRD) is correct in its representation of the equations, variables, and coefficients described in the FHFA whitepaper. My verification confirms that the mathematical and algorithmic specifications in the PRD accurately reflect the methodology detailed in the source document"

### PoC Results
* The pandas implementation resides under impl-pandas/ directory.
  * See [CLAUDE_CODE_SESSION_PANDAS.md](CLAUDE_CODE_SESSION_PANDAS.md) for all prompts issued to Claude Code. A summary response to each prompt by Claude Code is also included.
  * See [impl-pandas/TEST_SUMMARY.md](impl-pandas/TEST_SUMMARY.md) for the test summary report.
  * See [impl-pandas/README.md](impl-pandas/README.md) details on the Pandas implementation.
  * Time took to implement all 7 phases: 3 hours

### Running the Generated Code
Refer to [impl-pandas/README.md](impl-pandas/README.md)
