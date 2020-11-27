import numpy
cimport numpy


cdef class BeamEntry:
    """
    information about one single beam at specific time-step
    """
    cdef float alpha, beta
    cdef double p_total, p_non_blank, p_blank, p_lm
    cdef bint is_lm_applied
    cdef tuple labeling

    def __init__(self, float alpha, float beta):
        self.alpha = alpha
        self.beta = beta

        self.p_total = 0.0  # blank and non-blank
        self.p_non_blank = 0.0  # non-blank
        self.p_blank = 0.0  # blank
        self.p_lm = 1.0  # LM score
        self.is_lm_applied = False  # flag if LM was already applied to this beam
        self.labeling = ()  # beam-labeling

    property p_total:
        def __get__(self): return self.p_total
        def __set__(self, double other): self.p_total = other

    property p_non_blank:
        def __get__(self): return self.p_non_blank
        def __set__(self, double other): self.p_non_blank = other

    property p_blank:
        def __get__(self): return self.p_blank
        def __set__(self, double other): self.p_blank = other

    property p_lm:
        def __get__(self): return self.p_lm
        def __set__(self, double other): self.p_lm = other

    property is_lm_applied:
        def __get__(self): return self.is_lm_applied
        def __set__(self, bint other): self.is_lm_applied = other

    property labeling:
        def __get__(self): return self.labeling
        def __set__(self, tuple other): self.labeling = other

    @property
    def score(self):
        return self.p_total * (self.p_lm ** self.alpha) * (len(self.labeling) ** self.beta)


cdef class BeamState:
    """
    information about the beams at specific time-step
    """
    cdef float alpha, beta
    cdef dict entries

    def __init__(self, float alpha,  float beta):
        self.alpha = alpha
        self.beta = beta
        self.entries = {}

    @property
    def sorted_entries(self):
        return sorted(self.entries.values(), key=lambda x: x.score, reverse=True)

    def add_beam(self, labeling):
        if labeling not in self.entries:
            self.entries[labeling] = BeamEntry(alpha=self.alpha, beta=self.beta)


cdef apply_lm(BeamEntry beam, dict id2char, lm):
    """
    calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars
    """
    if lm and not beam.is_lm_applied:
        text = ''.join(id2char[i] for i in beam.labeling)
        beam.p_lm = lm(text)
        beam.is_lm_applied = True  # only apply LM once per beam entry


# ctypedef numpy.float_t DTYPE_t


def ctc_beam_search(
        numpy.ndarray[float, ndim=2] mat,
        dict classes,
        lm,
        int beam_width = 10,
        float alpha = 1.0,
        float beta = 2.0,
        float min_char_prob = 0.001
):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."
    cdef int blankIdx, maxT, maxC, t, c

    blankIdx = len(classes)
    maxT = mat.shape[0]
    maxC = mat.shape[1]

    # initialise beam state
    last = BeamState(alpha=alpha, beta=beta)
    labeling = ()
    last.entries[labeling] = BeamEntry(alpha=alpha, beta=beta)
    last.entries[labeling].p_blank = 1.0
    last.entries[labeling].p_total = 1.0

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
                p_non_blank = last.entries[labeling].p_non_blank * mat[t, labeling[-1]]
            else:
                p_non_blank = 0

            # probability of paths ending with a blank
            p_blank = last.entries[labeling].p_total * mat[t, blankIdx]

            # add beam at current time-step if needed
            curr.add_beam(labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].p_non_blank += p_non_blank
            curr.entries[labeling].p_blank += p_blank
            curr.entries[labeling].p_total += p_blank + p_non_blank
            # beam-labeling not changed, therefore also LM score unchanged from:
            curr.entries[labeling].p_lm = last.entries[labeling].p_lm
            # LM already applied at previous time-step for this beam-labeling:
            curr.entries[labeling].is_lm_applied = True

            # extend current beam-labeling
            for c in range(maxC - 1):
                # pruning
                if mat[t, c] < min_char_prob:
                    continue

                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    p_non_blank = mat[t, c] * last.entries[labeling].p_blank
                else:
                    p_non_blank = mat[t, c] * last.entries[labeling].p_total

                # add beam at current time-step if needed
                curr.add_beam(new_labeling)

                # fill in data
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].p_non_blank += p_non_blank
                curr.entries[new_labeling].p_total += p_non_blank

                # apply LM
                apply_lm(beam=curr.entries[new_labeling], id2char=classes, lm=lm)

        # set new beam state
        last = curr

    return last
