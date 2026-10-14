"""Format deterministic insights and investigation priorities for report output."""

from src.reporting.formatting import section, subsection, format_value


def insights_report(insights: dict):
    """
    Format the dictionary of insights into formatted structured text
    """
    lines = []

    lines.extend(
        section("3. COMPARISON INSIGHTS")
    )

    sections = [
        ("Network", insights.get("network", [])),
        ("System", insights.get("system", [])),
        ("Voltage", insights.get("voltage", [])),
        (
            "Generation / Load",
            insights.get("generation_load", []),
        ),
        (
            "Branch Flows",
            insights.get("branch_flows", []),
        ),
    ]

    for title, messages in sections:

        lines.extend(
            subsection(title)
        )

        lines.append("")

        if not messages:

            lines.append(
                "No additional insights identified."
            )

        else:

            for message in messages:

                lines.append(
                    f"• {message}"
                )

        lines.append("")

    return "\n".join(lines)


def statistical_insights_report(statistical_insights: dict):
    """Format statistical insights as a separate report section."""
    lines = []

    lines.extend(section("4. STATISTICAL INSIGHTS"))

    sections = [
        ("Overall Comparison Patterns", statistical_insights.get("overall", [])),
        ("Model Comparison", statistical_insights.get("model", [])),
        ("Power-Flow Comparison", statistical_insights.get("power_flow", [])),
    ]

    for title, findings in sections:
        lines.extend(subsection(title))
        lines.append("")

        if not findings:
            lines.append("No statistical findings identified.")
        else:
            for finding in findings:
                if isinstance(finding, dict):
                    # Accommodate dictionary-based findings.
                    description = (
                        finding.get("description")
                        or finding.get("message")
                        or str(finding)
                    )
                else:
                    description = str(finding)

                lines.append(f"• {description}")

        lines.append("")

    return "\n".join(lines)


def investigation_priorities_report(investigation_priorities: dict):
    """Format investigation priorities insights as a separate report section."""
    lines = []

    lines.extend(section("5. INVESTIGATION PRIORITIES"))

    lines.append("")

    priorities = investigation_priorities

    if not priorities:

        lines.append(
            "No severe differences requiring investigation "
            "were identified."
        )

    else:

        for rank, (
            asset,
            description,
            magnitude,
        ) in enumerate(priorities, start=1):

            description_text = str(description)
            if "model parameter mismatch" in description_text.lower():
                if "maximum mismatch" in description_text.lower():
                    priority_text = f"{rank}. {asset} — {description_text}."
                elif magnitude is not None:
                    priority_text = (
                        f"{rank}. {asset} — {description_text}; "
                        f"maximum mismatch of {format_value(magnitude)}."
                    )
                else:
                    priority_text = f"{rank}. {asset} — {description_text}."
            else:
                priority_text = (
                    f"{rank}. {asset} — {description_text} difference "
                    f"({format_value(magnitude)})."
                )
            lines.append(priority_text)

    lines.append("")

    return "\n".join(lines)
