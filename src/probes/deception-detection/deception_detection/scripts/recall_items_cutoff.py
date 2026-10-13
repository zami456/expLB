"""Question pairs for the after-cutoff pilot of the knowledge direction (`k_cutoff.py`).

K's unknowable questions (`recall_items.py`, `recall_items_matched.py`) ask for details nobody could
know. Here the unknown side is an answer that exists but the model does not have: each pair is one
question asked about two years. The earlier year's event happened before Llama-3.3's knowledge cutoff
(December 2023); the 2024 one happened after it but before the date in the chat template's header
(26 Jul 2024), so to the model it is a past event it was never told about. The two questions differ
in the year alone.

Each entry is (question with "{year}", known year, aliases of the known answer, aliases of the 2024
answer). `k_cutoff.py verify` keeps a pair only if the model names the known answer and does not name
the 2024 one; pairs are taken in this order.
"""

PAIRS = [
    ("Who won the women's singles title at Wimbledon in {year}?", 2019,
     ["Halep"], ["Krejčíková"]),
    ("Which country won the Eurovision Song Contest in {year}?", 2019,
     ["Netherlands", "Dutch", "Holland", "Duncan Laurence"], ["Switzerland", "Swiss", "Nemo"]),
    ("Who won the presidential election in Senegal in {year}?", 2019,
     ["Macky Sall"], ["Faye", "Diomaye"]),
    ("Who was awarded the Abel Prize in mathematics in {year}?", 2019,
     ["Uhlenbeck"], ["Talagrand"]),
    ("Which film won the Palme d'Or at the Cannes Film Festival in {year}?", 2019,
     ["Parasite", "Bong Joon"], ["Anora", "Sean Baker"]),
    ("Who won the men's singles title at the Australian Open in {year}?", 2019,
     ["Djokovic"], ["Sinner"]),
    ("Who won the presidential election in Slovakia in {year}?", 2019,
     ["Čaputová"], ["Pellegrini"]),
    ("Who won the Pritzker Architecture Prize in {year}?", 2019,
     ["Isozaki"], ["Yamamoto"]),
    ("Which horse won the Grand National in {year}?", 2019,
     ["Tiger Roll"], ["I Am Maximus"]),
    ("Which novel won the Pulitzer Prize for Fiction in {year}?", 2019,
     ["Overstory", "Richard Powers"], ["Night Watch", "Jayne Anne Phillips"]),
    ("Who won the presidential election in Iran in {year}?", 2021,
     ["Raisi", "Raeisi"], ["Pezeshkian"]),
    ("Who won the men's race of the London Marathon in {year}?", 2019,
     ["Kipchoge"], ["Mutiso", "Munyao"]),
    ("Who won the chess Candidates Tournament in {year}?", 2018,
     ["Caruana"], ["Gukesh"]),
    ("Who won the presidential election in Finland in {year}?", 2018,
     ["Niinistö"], ["Stubb"]),
    ("Who won the Monaco Grand Prix in Formula One in {year}?", 2019,
     ["Hamilton"], ["Leclerc"]),
    ("Which book won the International Booker Prize in {year}?", 2019,
     ["Celestial Bodies", "Alharthi"], ["Kairos", "Erpenbeck"]),
    ("Which club won the FA Cup final in {year}?", 2019,
     ["Manchester City", "Man City"], ["Manchester United", "Man United", "Man Utd"]),
    ("Who won the World Snooker Championship in {year}?", 2019,
     ["Judd Trump"], ["Kyren Wilson"]),
    ("Which horse won the Kentucky Derby in {year}?", 2018,
     ["Justify"], ["Mystik Dan"]),
    ("Which club won the UEFA Europa League final in {year}?", 2019,
     ["Chelsea"], ["Atalanta"]),
    ("Which country won the Africa Cup of Nations final played in {year}?", 2019,
     ["Algeria"], ["Ivory Coast", "Côte d'Ivoire"]),
    ("Which club won the German Bundesliga title in the season that ended in {year}?", 2019,
     ["Bayern"], ["Leverkusen"]),
    ("Who won the Australian Grand Prix in Formula One in {year}?", 2019,
     ["Bottas"], ["Sainz"]),
    ("Who won the men's race of the Boston Marathon in {year}?", 2018,
     ["Kawauchi"], ["Lemma"]),
    ("Who won the Giro d'Italia cycling race in {year}?", 2019,
     ["Carapaz"], ["Pogačar"]),
    ("Who won the men's title at Nathan's Hot Dog Eating Contest in {year}?", 2019,
     ["Chestnut"], ["Bertoletti"]),
    ("Who won the Academy Award for Best Actress at the ceremony held in {year}?", 2019,
     ["Colman"], ["Emma Stone"]),
    ("Who won the Masters golf tournament in {year}?", 2019,
     ["Tiger Woods", "Woods"], ["Scheffler"]),
    ("Who won the presidential election in Taiwan in {year}?", 2020,
     ["Tsai"], ["Lai Ching-te", "William Lai", "Lai"]),
    ("Who won the presidential election in Indonesia in {year}?", 2019,
     ["Widodo", "Jokowi"], ["Prabowo"]),
    ("Which team won the Stanley Cup in {year}?", 2019,
     ["St. Louis Blues", "Blues"], ["Florida Panthers", "Panthers"]),
    ("Who won the Women's Prize for Fiction in {year}?", 2019,
     ["Tayari Jones", "An American Marriage"], ["Ganeshananthan", "Brotherless Night"]),
    ("Who won the PDC World Darts Championship final played in {year}?", 2019,
     ["van Gerwen"], ["Humphries"]),
    ("Which film won the Golden Bear at the Berlin International Film Festival in {year}?", 2019,
     ["Synonyms", "Synonymes", "Nadav Lapid"], ["Dahomey", "Mati Diop"]),
]

UNKNOWN_YEAR = 2024
