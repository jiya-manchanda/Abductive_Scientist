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

    def explain(self, target, depth=3):
        """Backward chaining: find causes of a concept"""
        paths = []
        self._trace_explanation(target, [], paths, depth)
        return paths

    def _trace_explanation(self, target, current_path, all_paths, depth):
        if depth == 0:
            return
        for premise, conclusion in self.rules:
            if conclusion == target:
                new_path = [(premise, conclusion)] + current_path
                all_paths.append(new_path)
                self._trace_explanation(premise, new_path, all_paths, depth - 1)

    def predict(self, start, depth=3):
        """Forward chaining: find effects of a concept"""
        paths = []
        self._trace_prediction(start, [], paths, depth)
        return paths

    def _trace_prediction(self, current, current_path, all_paths, depth):
        if depth == 0:
            return
        for premise, conclusion in self.rules:
            if premise == current:
                new_path = current_path + [(premise, conclusion)]
                all_paths.append(new_path)
                self._trace_prediction(conclusion, new_path, all_paths, depth - 1)