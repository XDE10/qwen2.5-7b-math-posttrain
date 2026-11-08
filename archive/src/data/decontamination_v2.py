"""Read-only exact and lossless character-5gram Jaccard candidate screening."""
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = text.replace('\\left', '').replace('\\right', '').replace('\\!', '')
    return ' '.join(re.findall(r'\w+|[^\w\s]', text))


def exact(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def grams(text):
    value = normalize(text)
    return frozenset(value[i:i+5] for i in range(max(1, len(value)-4)))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


class Index:
    """Size filter + prefix filter yields no false negatives for threshold>0.

    A qualifying document must match at least one of the first
    len(query)-ceil(threshold*len(query))+1 query grams (any global ordering).
    Verify full Jaccard for all retained candidates. Rare grams first reduces work.
    """
    def __init__(self, rows):
        self.rows = rows
        self.sets = [grams(r['problem']) for r in rows]
        self.postings = defaultdict(list)
        for i, values in enumerate(self.sets):
            for value in values:
                self.postings[value].append(i)

    def query(self, text, threshold):
        import math
        q = grams(text)
        prefix = sorted(q, key=lambda value: (len(self.postings.get(value, ())), value))[:len(q)-math.ceil(threshold*len(q))+1]
        candidates = set()
        for value in prefix:
            candidates.update(self.postings.get(value, ()))
        found = []
        for i in candidates:
            d = self.sets[i]
            if not threshold*len(q) <= len(d) <= len(q)/threshold:
                continue
            similarity = jaccard(q, d)
            if similarity >= threshold:
                found.append((i, similarity))
        return sorted(found)


def calibrate(rows, paired_rows=None):
    # Training-side controlled positives, never tune thresholds on final tests.
    items = rows[:50]
    paired = {r['sample_id']:r['problem'] for r in paired_rows} if paired_rows else {}
    pairs = []
    for i, row in enumerate(items):
        text = row['problem']
        pairs.extend([
            ('known_identity', text, paired.get(row['sample_id'],text), True),
            ('synthetic_rewrite', text, 'Solve this mathematical problem. ' + text + ' Explain your answer.', True),
            ('numeric_family', text, re.sub(r'\d+', lambda m: str(int(m[0])+1), text), True),
            ('unrelated', text, rows[(i+347) % len(rows)]['problem'], False),
        ])
    scores = [(kind, jaccard(grams(a), grams(b)), positive) for kind, a, b, positive in pairs]
    grid = []
    for threshold in (.3, .4, .5, .6, .7, .8, .85, .9, .95):
        by_kind = {kind: sum(s >= threshold for k, s, _ in scores if k == kind)/50 for kind in {k for k, _, _ in scores}}
        grid.append({'threshold': threshold, 'recall': by_kind, 'positive_recall': sum(s >= threshold for _, s, p in scores if p)/150, 'false_positive_rate': by_kind['unrelated']})
    eligible = [g for g in grid if g['positive_recall'] >= .95 and g['false_positive_rate'] <= .05]
    chosen = max(eligible, key=lambda g: g['threshold']) if eligible else min(grid, key=lambda g: g['threshold'])
    return {'threshold': chosen['threshold'], 'grid': grid, 'selected': chosen,
            'scope': '50 sealed raw/compact known same-task pairs plus controlled synthetic rewrites/numeric families; not independently annotated semantic recall',
            'numeric_family': 'review signal, not proof of duplicate mathematical answer',
            'calibration_ids': [r['sample_id'] for r in items]}
