import re
import matplotlib.pyplot as plt
from collections import defaultdict

vreg_pattern = re.compile(r'(v|a)(\d{1,3}|\[\d{1,3}:\d{1,3}\])', re.IGNORECASE)
cf_barrier_pattern = re.compile(r'\b(s_branch|s_cbranch_[a-z_]+|s_endpgm|s_barrier)\b', re.IGNORECASE)


def insert_barriers(asm_lines):
    new_lines = []
    barrier_indices = []

    for i, line in enumerate(asm_lines):
        new_lines.append(line)
        if cf_barrier_pattern.search(line):
            new_lines.append('// BASIC_BLOCK_BARRIER')
            barrier_indices.append(len(new_lines) - 1)

    return new_lines, barrier_indices


def get_separate_regs(reg):
    res = []
    reg = reg.split(' ')[0]  # get rid of something like 'offset: 16' for now
    if reg.startswith("-"):
        reg = reg[1:]
    reg_type = reg[0].lower()
    reg_value = reg[1:]

    if reg_value.startswith('[') and ':' in reg_value:
        try:
            start, end = map(int, reg_value.strip('[]').split(':'))
            for i in range(start, end + 1):
                res.append(f"{reg_type}{i}")
        except ValueError:
            raise ValueError(f"Invalid register range: {reg}")
    else:
        try:
            res.append(f"{reg_type}{int(reg_value)}")
        except ValueError:
            print(reg)
            raise ValueError(f"Invalid register: {reg}")

    return res


def parse_def_use(asm_lines):
    live_ranges_dict = defaultdict(list)  # reg -> (def_idx, use_idx)

    for idx, line in enumerate(asm_lines):
        tokens = line.split()
        if len(tokens) < 2:
            continue

        # Extract operands after mnemonic
        operands = re.sub(r';.*', '', ' '.join(tokens[1:])).strip().split(',')
        # Clean and normalize register names
        regs = [op.strip().lower() for op in operands if vreg_pattern.match(op.strip().lstrip("-"))]

        if not regs:
            continue

        for def_reg in get_separate_regs(regs[0]):
            live_ranges_dict[def_reg].append((idx, idx))

        for use_regs in regs[1:]:
            for use_reg in get_separate_regs(use_regs):
                if (len(live_ranges_dict[use_reg]) > 0):
                    live_ranges_dict[use_reg][-1] = (live_ranges_dict[use_reg][-1][0], idx)
    live_ranges = [(key, a, b) for key, pairs in live_ranges_dict.items() for (a, b) in pairs]
    return live_ranges


def compute_live_counts(live_ranges, total_instrs):
    live_counts = [0] * total_instrs
    for _, start, end in live_ranges:
        for i in range(start, end + 1):
            if i < total_instrs:
                live_counts[i] += 1
    return live_counts


def parse_instr_count(asm_lines):
    instr_count_dict = defaultdict(int)  # token -> count
    for line in asm_lines:
        instr_count_dict[line.split()[0]] += 1
    return sorted(instr_count_dict.items(), key=lambda x: (-x[1], x[0]))


def main(filepath):
    with open(filepath, 'r') as f:
        original_lines = [line.strip() for line in f if line.strip()]

    asm_lines, barrier_indices = insert_barriers(original_lines)
    live_ranges = parse_def_use(asm_lines)
    live_counts = compute_live_counts(live_ranges, len(asm_lines))

    instr_count = parse_instr_count(asm_lines)

    print(f"{'Instr#':>7} | {'Live VRegs':>10} | Instruction")
    print('-' * 80)
    for i, (count, instr) in enumerate(zip(live_counts, asm_lines)):
        print(f"{i:7} | {count:10} | {instr}")

    print(f"{'Instruction':<25} Count")
    print("-" * 35)
    for instr, count in instr_count:
        print(f"{instr:<25} {count}")

    plt.figure(figsize=(14, 5))
    plt.plot(range(len(live_counts)), live_counts, label='Live vector registers', linewidth=1)
    for idx, barrier in enumerate(barrier_indices):
        plt.axvline(x=barrier, color='red', linestyle='--', linewidth=0.7,
                    label='Basic block barrier' if idx == 0 else "")

    plt.xlabel('Instruction Index')
    plt.ylabel('Live Vector Registers')
    plt.title('Vector Register Pressure with Basic Block Barriers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_register_pressure.py <asm_file>")
        sys.exit(1)
    main(sys.argv[1])
