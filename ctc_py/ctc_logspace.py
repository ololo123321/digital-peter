import numpy as np


NEG_INF = -np.inf


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = np.log10(sum(10 ** (a - a_max) for a in args))
    res = a_max + lsp
    # assert res != NEG_INF, args
    # assert res <= 0, f"res: {res}; args: {args}"
    return res


class ReprMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith('_'))
        return f'{class_name}({params})'


class BeamEntry(ReprMixin):
    """
    information about one single beam at specific time-step
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        self.p_total = NEG_INF  # blank and non-blank (log(0) = -inf)
        self.p_non_blank = NEG_INF  # non-blank (log(0) = -inf)
        self.p_blank = NEG_INF  # blank (log(0) = -inf)
        self.p_lm = 0  # LM score (log(1) = 0)
        self.is_lm_applied = False  # flag if LM was already applied to this beam
        self.labeling = ()  # beam-labeling

    @property
    def score(self):
        return self.p_total + self.alpha * self.p_lm + self.beta * np.log10(len(self.labeling) + 1)


class BeamState(ReprMixin):
    """
    information about the beams at specific time-step
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.entries = {}

    @property
    def sorted_entries(self):
        return sorted(self.entries.values(), key=lambda x: x.score, reverse=True)

    def add_beam(self, labeling):
        if labeling not in self.entries:
            self.entries[labeling] = BeamEntry(alpha=self.alpha, beta=self.beta)


def apply_lm(beam, id2char, lm):
    """
    calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars
    """
    if lm and not beam.is_lm_applied:
        text = ''.join(id2char[i] for i in beam.labeling)
        beam.p_lm = lm(text)
        beam.is_lm_applied = True  # only apply LM once per beam entry


def ctc_beam_search(mat, classes, lm, beam_width=10, alpha=1.0, beta=2.0, min_char_prob=0.001):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."
    mat_log = np.log10(mat)  # TODO: случай нулей
    # print(mat_log)

    blankIdx = len(classes)
    maxT, maxC = mat_log.shape

    # initialise beam state
    last = BeamState(alpha=alpha, beta=beta)
    labeling = ()
    last.entries[labeling] = BeamEntry(alpha=alpha, beta=beta)
    last.entries[labeling].p_blank = 0  # log(1) = 0
    last.entries[labeling].p_total = 0  # log(1) = 0

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState(alpha=alpha, beta=beta)

        # get beam-labelings of best beams
        best_entries = last.sorted_entries[:beam_width]

        # go over best beams
        for entry in best_entries:
            labeling = entry.labeling

            # probability of paths ending with a non-blank
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                p_non_blank = last.entries[labeling].p_non_blank + mat_log[t, labeling[-1]]
            else:
                p_non_blank = NEG_INF  # log(0) = -inf
            # assert p_non_blank <= 0

            # probability of paths ending with a blank
            p_blank = last.entries[labeling].p_total + mat_log[t, blankIdx]
            # assert p_blank <= 0
            # assert p_blank != NEG_INF, f"{t}, {labeling}, {last.entries[labeling].p_total}, {mat_log[t, blankIdx]}"

            # add beam at current time-step if needed
            curr.add_beam(labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].p_non_blank = logsumexp(curr.entries[labeling].p_non_blank, p_non_blank)
            curr.entries[labeling].p_blank = logsumexp(curr.entries[labeling].p_blank, p_blank)
            curr.entries[labeling].p_total = logsumexp(curr.entries[labeling].p_total, p_blank, p_non_blank)
            # beam-labeling not changed, therefore also LM score unchanged from:
            curr.entries[labeling].p_lm = last.entries[labeling].p_lm
            # LM already applied at previous time-step for this beam-labeling:
            curr.entries[labeling].is_lm_applied = True

            # assert curr.entries[labeling].p_non_blank <= 0
            # assert curr.entries[labeling].p_blank <= 0
            # assert curr.entries[labeling].p_total <= 0
            # assert curr.entries[labeling].p_lm <= 0

            # extend current beam-labeling
            for c in range(maxC - 1):
                # pruning
                if mat[t, c] < min_char_prob:
                    continue

                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    p_non_blank = mat_log[t, c] + last.entries[labeling].p_blank
                else:
                    p_non_blank = mat_log[t, c] + last.entries[labeling].p_total

                # add beam at current time-step if needed
                curr.add_beam(new_labeling)

                # fill in data
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].p_non_blank = logsumexp(curr.entries[new_labeling].p_non_blank, p_non_blank)
                curr.entries[new_labeling].p_total = logsumexp(curr.entries[new_labeling].p_total, p_non_blank)

                # apply LM
                apply_lm(beam=curr.entries[new_labeling], id2char=classes, lm=lm)

        # set new beam state
        last = curr

    # res = {}
    # items_sorted = sorted(last.entries.items(), reverse=True, key=lambda x: x[1].p_total * x[1].p_lm)
    # for labeling, entry in items_sorted[:top_paths]:
    #     s = ''
    #     for l in labeling:
    #         s += classes[l]
    #     res[s] = entry.p_total * entry.p_lm
    #
    # return res

    return last


def test_beam_search():
    classes = 'ab'
    mat = np.array([
        [0.4, 0, 0.6],
        [0.4, 0, 0.6]
    ])
    print('Test beam search')
    expected = 'a'
    res = ctc_beam_search(
        mat=mat,
        classes=classes,
        lm=None,
        alpha=0,
        beta=0
    )
    for beam in res.entries.values():
        assert beam.p_lm <= 0, beam.p_lm
        assert beam.p_blank <= 0, beam.p_blank
        assert beam.p_non_blank <= 0, beam.p_non_blank
        assert beam.p_total <= 0, beam.p_total
    print(res)
    best_entry = res.sorted_entries[0]
    print(best_entry)
    actual = ''.join(classes[i] for i in best_entry.labeling)
    print('Expected: "' + expected + '"')
    print('Actual: "' + actual + '"')
    print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
    test_beam_search()
