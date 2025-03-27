class SymbolicReasoner:
    def __init__(self, rules_path):
        self.rules = self.load_rules(rules_path)

    def load_rules(self, path):
        rules = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=>' in line:
                    parts = line.split('=>')
                    premise = parts[0].strip()
                    conclusion = parts[1].strip()
                    rules.append((premise, conclusion))
        return rules

    def explain(self, target_concept, depth=3):
        paths = []
        self._trace_chain(target_concept, [], paths, depth)
        return paths

    def _trace_chain(self, target, current_path, all_paths, depth):
        if depth == 0:
            return
        for premise, conclusion in self.rules:
            if conclusion == target:
                new_path = [(premise, conclusion)] + current_path
                all_paths.append(new_path)
                self._trace_chain(premise, new_path, all_paths, depth - 1)