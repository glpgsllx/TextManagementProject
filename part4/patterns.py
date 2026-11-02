"""Regular expressions that define what a comparison looks like.

Every family is a way English marks a comparison between two things. The
families are kept apart so the report can say *why* a sentence was flagged, and
so the evaluation can show which family contributes the errors.
"""

import re

# Adjectives and adverbs that take "than" when they compare two things.
COMPARATIVE = (
    r"(?:better|worse|cheaper|pricier|faster|quicker|slower|friendlier|nicer|"
    r"cleaner|dirtier|tastier|fresher|bigger|smaller|larger|greater|higher|"
    r"lower|superior|inferior|easier|harder|busier|quieter|louder|warmer"
    r")"
)

FAMILIES = {
    # "better than the place on Main", "cheaper than Panera"
    "than": re.compile(rf"\b{COMPARATIVE}\b[^.;!?]{{0,60}}\bthan\b"),
    # "more options than", "less wait than", "fewer tables than"
    "more_less_than": re.compile(r"\b(?:more|less|fewer)\b[^.;!?]{0,40}\bthan\b"),
    # "compared to the one downtown", "versus", "vs."
    "compared": re.compile(r"\bcompared (?:to|with)\b|\bin comparison (?:to|with)\b|\bversus\b|\bvs\.?\b"),
    # "same as", "as good as", "similar to", "comparable to"
    "same_as": re.compile(r"\b(?:same as|as good as|as bad as|as nice as|comparable to|similar to)\b"),
    # "beats the one on Eagle Road", "blows it away", "out of the water"
    "beat": re.compile(r"\b(?:beats?|beat|blew .{0,25} away|blows? .{0,25} away|out of the water|runs? circles)\b"),
    # "I prefer X over Y", "we chose this over the chain"
    "prefer": re.compile(r"\bprefer\b.{0,40}\b(?:over|to)\b|\bchose\b.{0,30}\bover\b"),
    # "unlike the other places in town"
    "unlike": re.compile(r"\bunlike\b"),
    # "the best pizza in town", "the worst meal I've ever had"
    "superlative": re.compile(
        r"\b(?:best|worst|greatest|finest)\b[^.;!?]{0,60}\b(?:in town|around here|"
        r"in the (?:area|valley|city|state)|i(?:'ve| have) (?:ever )?(?:had|been|tried|visited|seen)|anywhere)\b"
    ),
}

# Sentences that contain a comparison word but not a comparison of two
# businesses. These are idioms, quantities and comparisons with an expectation.
NON_BUSINESS = re.compile(
    r"\b(?:"
    r"other than|rather than|"
    r"more than (?:happy|willing|glad|accommodating|enough|welcome|once|a few|a little|just|double|half|a couple|i can say)|"
    r"less than (?:an? )?(?:hour|minute|day|week|month|year|\$|\d)|"
    r"more than (?:an? )?(?:hour|minute|day|week|month|year|\$|\d)|"
    r"much more than a[n]? \w+|more than just a[n]? \w+|"
    r"better than (?:nothing|ever|expected|i thought|new|before|most|average)|"
    r"better late than never|nothing less than|less than impressed|"
    r"than (?:i|we|you) (?:thought|expected|imagined)|no more than|no less than"
    r")\b"
)

# A comparison target that names or implies another business without giving its
# name: "the one downtown", "other places", "competitors", "the chain".
PLACE_TARGET = re.compile(
    r"\b(?:(?:the|that|this) (?:one|place) (?:in|on|at|near|downtown|across|down)|"
    r"other (?:\w+ ){0,2}(?:places?|restaurants?|businesses?|shops?|salons?|hotels?|"
    r"bars?|cafes?|stores?|parks?|gyms?|clinics?|spas?|vets?|studios?|dealerships?|"
    r"cleaners?|laundromats?|theaters?|banks?|apartments?)|"
    r"competitors?|chains?|other locations?|downtown|across town|"
    r"in (?:boise|meridian|nampa|eagle|caldwell|kuna|star))\b"
)

# A generic hospitality noun in the target phrase: "better than the restaurant
# down the street" is a business comparison even without a name.
BUSINESS_NOUN = re.compile(
    r"\b(?:restaurants?|places?|pizzerias?|cafes?|coffee shops?|salons?|shops?|"
    r"stores?|hotels?|motels?|bars?|grills?|diners?|bakeries?|competitors?|"
    r"chains?|locations?|establishments?|parks?|gyms?|clinics?|spas?|vets?|"
    r"veterinarians?|studios?|dealerships?|cleaners?|laundromats?|theaters?|"
    r"banks?|dentists?|doctors?|offices?|apartments?|complexes?)\b"
)
