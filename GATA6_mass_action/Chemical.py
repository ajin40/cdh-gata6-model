class Chemical:

    """Chemical is a class describing chemicals and/or chemical complexes.
    Chemical.amount: the amount of the Chemical in the cell.
    Chemical.reactions: a list of reactions whose rates are changed if
    Chemical.amount is altered."""

    def __init__(self, amount):
        # Chemical.amount represents the amount of this Chemical in the cell
        self.amount = amount
        # Chemical.reactions is a list of reactions whose rates are changed if
        # Chemical.amount is altered.
        self.reactions = []


class DegradationReaction:

    """DegradationReaction describes the removal of one molecule of
    the specified substrate, with specified rate_constant.
    Overall rate for this reaction to occur is
    substrate.amount * rate_constant."""

    def __init__(self, substrate, rate_constant):
        self.stoichiometry = {substrate: -1}
        self.substrate = substrate
        self.rate_constant = rate_constant
        substrate.reactions.append(self)

    def GetRate(self):
        return self.substrate.amount * self.rate_constant


class CatalyzedSynthesisReaction:

    """CatalyzedSynthesisReaction describes the synthesis of product in'
    the presence of a catalyst:  catalyst -> catalyst + product,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    catalyst.amount * rate_constant."""

    def __init__(self, catalyst, product, rate_constant):
        self.stoichiometry = {catalyst: 0, product: 1}
        self.catalyst = catalyst
        self.rate_constant = rate_constant
        product.reactions.append(self)

    def GetRate(self):
        return self.catalyst.amount * self.rate_constant


class HeterodimerBindingReaction:

    """HeterodimerBindingReaction describes the binding of two distinct
    types of chemicals, A and B, to form a product dimer: A + B -> dimer,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    A.amount * B.amount * rate_constant."""

    def __init__(self, A, B, dimer, rate_constant):
        self.stoichiometry = {A: -1, B: -1, dimer: 1}
        self.A = A
        self.B = B
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        dimer.reactions.append(self)

    def GetRate(self):
        return self.A.amount * self.B.amount * self.rate_constant


class HeterodimerUnbindingReaction:

    """HeterodimerBindingReaction describes the unbinding of a
    heterodimer into two distinct types of chemicals, A and B:
    dimer -> A + B, with specified rate_constant.
    Overall rate for this reaction to occur is
    dimer.amount * rate_constant."""

    def __init__(self, dimer, A, B, rate_constant):
        self.stoichiometry = {A: 1, B: 1, dimer: -1}
        self.dimer = dimer
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        dimer.reactions.append(self)

    def GetRate(self):
        return self.dimer.amount * self.rate_constant

# ---------------------------------------------------------------------
