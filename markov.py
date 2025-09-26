import argparse
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

START_STATE = "start"
CONVERSION_STATE = "conversion"
ABANDON_STATE = "abandon"
ABSORBING_STATES = {CONVERSION_STATE, ABANDON_STATE}


def canonicalize_state(state: str) -> str:
    cleaned = state.strip()
    if cleaned.startswith("//"):
        cleaned = "/" + cleaned.lstrip("/")
    return cleaned


POST_CONVERSION_PATHS_RAW = [
    "/checkout",
    "/cadastro/checkout",
    "/onboarding",
    "/products",
    "/invoices",
    "/cadastro",
    "/legal",
    "/home",
    "/settings",
    "/select-account",
    "/app-infinitepay",
    "/transactions",
    "/plans",
    "/impersonate",
    "/statements",
    "/receivables",
    "/action",
    "/plano",
    "/dados-pessoais",
    "//dados-pessoais",
    "/email-confirmation",
    "/create",
    "/edit",
    "/escolher-plano",
    "/personalize",
    "/v2",
    "/validar-identidade",
    "/your-infinitepay",
    "/review",
    "/contrato-de-afiliacao",
    "/add",
    "/dados-empresa",
    "/profile",
    "/pos-orders",
    "/success",
    "/security",
    "/change",
    "/termos-de-uso",
    "/team-management",
    "/cadastrar-novo-cnpj",
    "/entrega",
    "/conta-ja-registrada",
    "/erro",
    "/muitas-tentativas",
    "/cnae",
    "/politica-de-troca-devolucao",
    "/phone-number",
    "/desenvolvedores",
    "/confirmation",
    "/add-accountant",
    "/add-member",
    "/payment-methods",
    "/politica-de-protecao-de-dados-pessoais",
    "/sucesso",
    "/produto-adquirido",
    "/acesse-o-app",
    "/endereco",
    "/cnpj",
    "/new",
    "/social-medias",
    "/cards",
    "/clients",
    "/product",
    "/lending",
    "/reports",
    "/add-base-url",
    "/add-customers",
    "/pessoa-fisica",
    "/link/pessoa-fisica",
    "/link/tap",
    "/link/link-de-pagamento",
    "/link/maquina-cartao",
    "/link/maquininha",
    "/link/loja-online",
    "/link/taxas",
    "/link/referral",
    "/smart",
]

POST_CONVERSION_PATHS = {canonicalize_state(path) for path in POST_CONVERSION_PATHS_RAW}


def is_id_like_channel(channel: str) -> bool:
    if channel in ABSORBING_STATES or channel == START_STATE:
        return False
    if "checkout" in channel:
        return False
    trimmed = channel.strip("/")
    if not trimmed:
        return False
    if not any(char.isdigit() for char in trimmed):
        return False
    alnum = "".join(char for char in trimmed if char.isalnum())
    return len(alnum) >= 3


EXTRA_SKIP_PATHS = {
    "/faturamento",
    "/external-checkout",
    "/devices",
}

EXTRA_SKIP_SUBSTRINGS = ("/link/",)


def should_skip_channel(channel: str) -> bool:
    if channel in POST_CONVERSION_PATHS:
        return True
    if channel in EXTRA_SKIP_PATHS:
        return True
    if any(fragment in channel for fragment in EXTRA_SKIP_SUBSTRINGS):
        return True
    return is_id_like_channel(channel)


PARENT_PREFIX_CHANNELS = {
    "/blog",
    "/materiais",
    "/lp",
    "/link",
    "/gestao-cobranca",
    "/gestao-de-cobranca",
    "/help-center",
    "/sac",
}


SKIP_COMBINED_PATHS = {
    "/link/link-na-bio",
    "/link/lp",
    "/link/link-de-pagamento",
    "/link/tap",
    "/link/maquina-cartao",
    "/link/maquininha",
    "/link/taxas",
    "/link/loja-online",
    "/link/referral",
    "/link/pessoa-fisica",
    "/smart",
}


def parse_sequences(path: str) -> Tuple[List[List[str]], List[bool]]:
    sequences: List[List[str]] = []
    conversions: List[bool] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "markov_seq" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'markov_seq' column")
        if "converted" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'converted' column")
        for row in reader:
            raw_seq = row["markov_seq"]
            if not raw_seq:
                continue
            steps = [canonicalize_state(part) for part in raw_seq.split(">") if part.strip()]
            if not steps:
                continue
            if steps[0] != START_STATE:
                steps.insert(0, START_STATE)
            if steps[-1] not in ABSORBING_STATES:
                raise ValueError(f"Sequence does not terminate in an absorbing state: {steps}")
            cleaned_steps: List[str] = []
            previous_kept: Optional[str] = None
            for state in steps:
                if state == START_STATE or state in ABSORBING_STATES:
                    cleaned_steps.append(state)
                    previous_kept = state
                    continue
                if should_skip_channel(state):
                    continue
                if state == "/infinitetap":
                    if not previous_kept or not previous_kept.startswith("/blog"):
                        continue
                    if previous_kept.endswith("/infinitetap"):
                        continue
                    joined = f"{previous_kept.rstrip('/')}/infinitetap"
                    new_state = canonicalize_state(joined)
                    cleaned_steps.append(new_state)
                    previous_kept = new_state
                    continue
                new_state = state
                if (
                    previous_kept
                    and previous_kept not in ABSORBING_STATES
                    and previous_kept != START_STATE
                    and previous_kept != "/"
                    and previous_kept in PARENT_PREFIX_CHANNELS
                    and state.startswith("/")
                    and state.lstrip("/").count("/") == 0
                    and state != "/"
                    and state != previous_kept
                    and not state.startswith(previous_kept.rstrip("/") + "/")
                ):
                    candidate = previous_kept.rstrip("/") + state
                    if candidate in SKIP_COMBINED_PATHS:
                        new_state = state
                    else:
                        new_state = canonicalize_state(candidate)
                cleaned_steps.append(new_state)
                previous_kept = new_state
            if len(cleaned_steps) < 2:
                continue
            if cleaned_steps[-1] not in ABSORBING_STATES:
                cleaned_steps.append(ABANDON_STATE)
            sequences.append(cleaned_steps)
            converted_flag = str(row["converted"]).strip().lower() in {"1", "true", "t", "yes", "y"}
            conversions.append(converted_flag)
    if not sequences:
        raise ValueError("No sequences were parsed from the CSV file")
    return sequences, conversions


def build_transitions(sequences: Sequence[Sequence[str]]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for steps in sequences:
        for current_state, next_state in zip(steps[:-1], steps[1:]):
            counts[current_state][next_state] += 1
    transitions: Dict[str, Dict[str, float]] = {}
    for state, counter in counts.items():
        total = sum(counter.values())
        if total == 0:
            continue
        transitions[state] = {next_state: value / total for next_state, value in counter.items()}
    for absorbing in ABSORBING_STATES:
        transitions.setdefault(absorbing, {})
    return transitions


def absorption_probability(transitions: Dict[str, Dict[str, float]], start_state: str) -> float:
    states = set(transitions.keys())
    for neighbours in transitions.values():
        states.update(neighbours.keys())
    values: Dict[str, float] = {state: 0.0 for state in states}
    for absorbing in ABSORBING_STATES:
        values[absorbing] = 1.0 if absorbing == CONVERSION_STATE else 0.0
    max_iter = 10000
    tolerance = 1e-12
    for _ in range(max_iter):
        delta = 0.0
        for state in states:
            if state in ABSORBING_STATES:
                continue
            neighbours = transitions.get(state, {})
            updated = 0.0
            for next_state, probability in neighbours.items():
                updated += probability * values.get(next_state, 0.0)
            delta = max(delta, abs(values[state] - updated))
            values[state] = updated
        if delta < tolerance:
            break
    return values.get(start_state, 0.0)


def collect_touch_stats(sequences: Sequence[Sequence[str]], conversions: Sequence[bool]) -> Tuple[Counter, Counter, Counter]:
    total = Counter()
    converted = Counter()
    not_converted = Counter()
    for steps, did_convert in zip(sequences, conversions):
        touches = steps[1:-1]  # drop start and absorbing state
        for touch in touches:
            total[touch] += 1
            if did_convert:
                converted[touch] += 1
            else:
                not_converted[touch] += 1
    return total, converted, not_converted


def compute_removal_effects(
    base_transitions: Dict[str, Dict[str, float]],
    channels: Sequence[str],
    base_conversion_probability: float,
) -> Dict[str, float]:
    effects: Dict[str, float] = {}
    if base_conversion_probability == 0.0:
        return {channel: 0.0 for channel in channels}
    for channel in channels:
        transitions = remove_channel_from_transitions(base_transitions, channel)
        new_probability = absorption_probability(transitions, START_STATE)
        diff = base_conversion_probability - new_probability
        effects[channel] = diff if diff > 0 else 0.0
    return effects


def remove_channel_from_transitions(
    transitions: Dict[str, Dict[str, float]], channel: str
) -> Dict[str, Dict[str, float]]:
    trimmed: Dict[str, Dict[str, float]] = {}
    for state, outcomes in transitions.items():
        if state == channel:
            continue
        if state in ABSORBING_STATES:
            trimmed[state] = {}
            continue
        updated: Dict[str, float] = {}
        for next_state, probability in outcomes.items():
            if next_state == channel:
                continue
            updated[next_state] = probability
        remaining = sum(updated.values())
        missing = 1.0 - remaining
        if missing > 1e-12:
            updated[ABANDON_STATE] = updated.get(ABANDON_STATE, 0.0) + missing
        elif missing < -1e-12:
            scale = 1.0 / remaining
            updated = {state_key: probability * scale for state_key, probability in updated.items()}
        trimmed[state] = updated
    for absorbing in ABSORBING_STATES:
        trimmed.setdefault(absorbing, {})
    return trimmed


def format_number(value: float) -> str:
    return f"{value:,.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov chain attribution for touchpoint sequences")
    parser.add_argument("input_csv", help="Path to the CSV file with markov_seq and converted columns")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path to export channel contributions as CSV",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=0,
        help="Only display the top N channels when printing results",
    )
    args = parser.parse_args()

    sequences, conversions = parse_sequences(args.input_csv)
    transitions = build_transitions(sequences)
    base_conversion_probability = absorption_probability(transitions, START_STATE)

    total_sequences = len(sequences)
    total_conversions = sum(1 for flag in conversions if flag)
    touch_total, touch_conv, touch_non_conv = collect_touch_stats(sequences, conversions)

    channels = sorted(touch_total.keys())
    removal_effects = compute_removal_effects(transitions, channels, base_conversion_probability)

    effect_sum = sum(removal_effects.values())
    results = []
    for channel in channels:
        effect = removal_effects[channel]
        share = effect / effect_sum if effect_sum > 0 else 0.0
        contribution = total_conversions * share
        results.append(
            {
                "channel": channel,
                "touches_total": touch_total[channel],
                "touches_converted": touch_conv[channel],
                "touches_not_converted": touch_non_conv[channel],
                "removal_effect": effect,
                "attributed_conversions": contribution,
                "attribution_share": share,
            }
        )

    results.sort(key=lambda item: item["attribution_share"], reverse=True)

    print("Base statistics")
    print(f"  Sequences: {total_sequences}")
    print(f"  Conversions: {total_conversions}")
    print(f"  Observed conversion rate: {total_conversions / total_sequences:.4f}")
    print(f"  Markov conversion probability: {base_conversion_probability:.4f}")
    print()

    headers = [
        "channel",
        "touches_total",
        "touches_converted",
        "touches_not_converted",
        "removal_effect",
        "attributed_conversions",
        "attribution_share",
    ]
    col_widths = {header: len(header) for header in headers}
    for row in results:
        for header in headers:
            value = row[header]
            text = format_number(value) if isinstance(value, float) else str(value)
            col_widths[header] = max(col_widths[header], len(text))

    limit = args.top if args.top and args.top > 0 else len(results)
    header_line = " | ".join(header.ljust(col_widths[header]) for header in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in results[:limit]:
        line_parts = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                text = format_number(value)
            else:
                text = str(value)
            line_parts.append(text.ljust(col_widths[header]))
        print(" | ".join(line_parts))

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in results:
                persisted = row.copy()
                persisted.update({
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                })
                writer.writerow(persisted)
        print(f"\nResults exported to {args.output}")


if __name__ == "__main__":
    main()
