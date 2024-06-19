import random
import itertools
import uuid
import re
from datasets import Dataset
from collections import defaultdict
from fastchat.model import get_conversation_template

def get_tokenized_len(entry: dict, tokenizer):
    question = entry["question"]
    # get tokenized length
    tokenized_len = len(tokenizer(question, return_tensors="pt")["input_ids"][0])

    return {"tokenized_len": tokenized_len}

def to_question(entry: dict) -> dict:
    """
    Convert a single entry to a “question”
    """
    prompt_header = "Below is a record of lines I want you to remember. " + \
                    "Each line begins with 'line <line index>' and contains " + \
                    "a '<REGISTER_CONTENT>' at the end of the line as a numerical value. " + \
                    "For each line index, memorize its corresponding <REGISTER_CONTENT>. " + \
                    "At the end of the record, I will ask you to retrieve the corresponding <REGISTER_CONTENT> of a certain line index. " + \
                    "Now the record start:\n\n"
    
    key_str = entry["key_str"]
    value = entry["value"]

    instruction = f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in line {key_str}? I need the number."

    user_input = prompt_header + entry["content"] + instruction

    return {"question" : user_input}

def to_single_round_conversation(entry: dict, model_name: str) -> dict:
    user_input = to_question(entry)["question"]

    key_str = entry["key_str"]
    value = entry["value"]
    model_response = f"The <REGISTER_CONTENT> in line {key_str} is <{value}>."

    conv = get_conversation_template(model_name)
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], model_response)
    conversation = conv.get_prompt()

    correct_answer = f"<{value}>"

    return {"conversation" : conversation}

def to_multi_round_conversation(entry: dict, model_dir: str, num_questions: int) -> dict:
    prompt_header = "Below is a record of lines I want you to remember. " + \
                    "Each line begins with 'line <line index>' and contains " + \
                    "a '<REGISTER_CONTENT>' at the end of the line as a numerical value. " + \
                    "For each line index, memorize its corresponding <REGISTER_CONTENT>. " + \
                    "At the end of the record, I will ask you to retrieve the corresponding <REGISTER_CONTENT> of a certain line index. " + \
                    "Now the record start:\n\n"
    
    content = entry["content"]
    num_lines = entry["num_lines"]

    # find all the key_str and value pairs in the content
    # example: "line new-distributor: REGISTER_CONTENT is <34884>\n"
    content_list = content.split("\n")[:-1]

    assert len(content_list) == num_lines
    
    instruction = f"\nNow the record is over. "

    conv = get_conversation_template(model_dir)

    key_ids = []

    for question_id in range(num_questions):
        key_id = random.randint(1, num_lines)
        key_ids.append(key_id)

        line = content_list[key_id - 1]
        key_str = re.search(r"line (.+?):", line).group(1)
        value = re.search(r"REGISTER_CONTENT is <(.+?)>", line).group(1)

        question = f"Tell me what is the <REGISTER_CONTENT> in line {key_str}? I need the number."
        model_response = f"The <REGISTER_CONTENT> in line {key_str} is <{value}>."

        if question_id == 0:
            user_input = prompt_header + entry["content"] + instruction + question
        else:
            user_input = question

        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], model_response)

    conversation = conv.get_prompt()
    return {"conversation" : conversation, "key_id_list": key_ids}
    
example = {
    "num_lines": 320,
    "content": 'line enchanting-jeweller: REGISTER_CONTENT is <21342>\nline cautious-chowder: REGISTER_CONTENT is <29286>\nline rhetorical-subject: REGISTER_CONTENT is <2190>\nline curly-external: REGISTER_CONTENT is <22548>\nline half-fiberglass: REGISTER_CONTENT is <29033>\nline low-error: REGISTER_CONTENT is <5630>\nline spectacular-exaggeration: REGISTER_CONTENT is <31065>\nline imported-fish: REGISTER_CONTENT is <18714>\nline nutty-robe: REGISTER_CONTENT is <17666>\nline miscreant-plugin: REGISTER_CONTENT is <28438>\nline lying-statin: REGISTER_CONTENT is <9586>\nline ugly-diagram: REGISTER_CONTENT is <47949>\nline boorish-lunch: REGISTER_CONTENT is <14207>\nline good-denim: REGISTER_CONTENT is <1274>\nline shallow-stadium: REGISTER_CONTENT is <4602>\nline woozy-rabbit: REGISTER_CONTENT is <7988>\nline earthy-reluctance: REGISTER_CONTENT is <15529>\nline rebellious-coast: REGISTER_CONTENT is <41584>\nline encouraging-squash: REGISTER_CONTENT is <37>\nline high-pitched-proprietor: REGISTER_CONTENT is <36151>\nline tested-creator: REGISTER_CONTENT is <26132>\nline woebegone-mind: REGISTER_CONTENT is <12112>\nline roomy-silo: REGISTER_CONTENT is <2295>\nline lovely-yogurt: REGISTER_CONTENT is <2235>\nline fuzzy-modem: REGISTER_CONTENT is <47108>\nline smoggy-spotlight: REGISTER_CONTENT is <34848>\nline impossible-poncho: REGISTER_CONTENT is <901>\nline short-weakness: REGISTER_CONTENT is <15965>\nline skinny-ghost: REGISTER_CONTENT is <1999>\nline lucky-sandwich: REGISTER_CONTENT is <20453>\nline calm-entity: REGISTER_CONTENT is <13397>\nline loose-terracotta: REGISTER_CONTENT is <37623>\nline moldy-disengagement: REGISTER_CONTENT is <17798>\nline worried-wardrobe: REGISTER_CONTENT is <33328>\nline decorous-fanlight: REGISTER_CONTENT is <28904>\nline pathetic-timing: REGISTER_CONTENT is <16251>\nline absorbing-lookout: REGISTER_CONTENT is <37397>\nline easy-tiger: REGISTER_CONTENT is <15405>\nline freezing-trapdoor: REGISTER_CONTENT is <13432>\nline ludicrous-vision: REGISTER_CONTENT is <7206>\nline various-dawn: REGISTER_CONTENT is <41728>\nline vague-spirit: REGISTER_CONTENT is <36436>\nline fast-toenail: REGISTER_CONTENT is <13535>\nline unadvised-fedelini: REGISTER_CONTENT is <37347>\nline outrageous-balcony: REGISTER_CONTENT is <33275>\nline agreeable-phenomenon: REGISTER_CONTENT is <13592>\nline precious-ram: REGISTER_CONTENT is <37733>\nline feigned-alphabet: REGISTER_CONTENT is <38677>\nline huge-lightscreen: REGISTER_CONTENT is <37374>\nline frail-vengeance: REGISTER_CONTENT is <25985>\nline young-obstacle: REGISTER_CONTENT is <12604>\nline ugliest-craftsman: REGISTER_CONTENT is <6130>\nline flaky-window: REGISTER_CONTENT is <40768>\nline idiotic-chateau: REGISTER_CONTENT is <30690>\nline pretty-forebear: REGISTER_CONTENT is <2417>\nline magnificent-record: REGISTER_CONTENT is <8605>\nline damp-shot: REGISTER_CONTENT is <27828>\nline psychotic-homicide: REGISTER_CONTENT is <13200>\nline adamant-continent: REGISTER_CONTENT is <3072>\nline alike-pyramid: REGISTER_CONTENT is <27677>\nline energetic-dam: REGISTER_CONTENT is <42675>\nline anxious-ex-wife: REGISTER_CONTENT is <44637>\nline chunky-metabolite: REGISTER_CONTENT is <23249>\nline nonstop-conspiracy: REGISTER_CONTENT is <37782>\nline forgetful-machine: REGISTER_CONTENT is <35859>\nline puffy-spark: REGISTER_CONTENT is <34586>\nline fragile-punch: REGISTER_CONTENT is <9395>\nline dull-patent: REGISTER_CONTENT is <24111>\nline kindhearted-rack: REGISTER_CONTENT is <38023>\nline rough-tortoise: REGISTER_CONTENT is <959>\nline noxious-cranberry: REGISTER_CONTENT is <16078>\nline chilly-variant: REGISTER_CONTENT is <22589>\nline cultured-calculus: REGISTER_CONTENT is <39707>\nline garrulous-blight: REGISTER_CONTENT is <16736>\nline magenta-fennel: REGISTER_CONTENT is <36565>\nline wistful-union: REGISTER_CONTENT is <12516>\nline eminent-shipyard: REGISTER_CONTENT is <43708>\nline bright-literature: REGISTER_CONTENT is <18117>\nline lackadaisical-anarchist: REGISTER_CONTENT is <1955>\nline unable-range: REGISTER_CONTENT is <30930>\nline loving-downturn: REGISTER_CONTENT is <34949>\nline funny-salsa: REGISTER_CONTENT is <48572>\nline hushed-pard: REGISTER_CONTENT is <222>\nline lewd-gravity: REGISTER_CONTENT is <41117>\nline historical-astrolabe: REGISTER_CONTENT is <21671>\nline excellent-handmaiden: REGISTER_CONTENT is <39742>\nline known-reader: REGISTER_CONTENT is <26462>\nline jobless-coral: REGISTER_CONTENT is <20028>\nline fresh-dirndl: REGISTER_CONTENT is <5555>\nline vast-overcoat: REGISTER_CONTENT is <43139>\nline petite-hutch: REGISTER_CONTENT is <36235>\nline offbeat-visa: REGISTER_CONTENT is <44905>\nline faulty-scientist: REGISTER_CONTENT is <19970>\nline adventurous-magazine: REGISTER_CONTENT is <29487>\nline ad hoc-rocket-ship: REGISTER_CONTENT is <5591>\nline warlike-muffin: REGISTER_CONTENT is <35610>\nline drunk-arm-rest: REGISTER_CONTENT is <5957>\nline materialistic-rage: REGISTER_CONTENT is <29934>\nline foolish-forte: REGISTER_CONTENT is <14276>\nline fine-gobbler: REGISTER_CONTENT is <31117>\nline hulking-charm: REGISTER_CONTENT is <22559>\nline plucky-dictionary: REGISTER_CONTENT is <49889>\nline elegant-berry: REGISTER_CONTENT is <14031>\nline cool-wombat: REGISTER_CONTENT is <15526>\nline daffy-ruffle: REGISTER_CONTENT is <43580>\nline evil-best-seller: REGISTER_CONTENT is <38194>\nline tender-pate: REGISTER_CONTENT is <37869>\nline unusual-coverage: REGISTER_CONTENT is <27575>\nline oval-industry: REGISTER_CONTENT is <1793>\nline tangy-limo: REGISTER_CONTENT is <48684>\nline charming-damage: REGISTER_CONTENT is <17338>\nline flipped-out-grab-bag: REGISTER_CONTENT is <41031>\nline puzzled-cabbage: REGISTER_CONTENT is <4000>\nline tawdry-overnighter: REGISTER_CONTENT is <41631>\nline imaginary-pair: REGISTER_CONTENT is <49418>\nline afraid-armchair: REGISTER_CONTENT is <2311>\nline green-shortage: REGISTER_CONTENT is <9228>\nline poor-insolence: REGISTER_CONTENT is <15514>\nline tenuous-manacle: REGISTER_CONTENT is <38748>\nline careful-shoehorn: REGISTER_CONTENT is <36689>\nline few-eyebrow: REGISTER_CONTENT is <38572>\nline learned-wrapper: REGISTER_CONTENT is <34324>\nline bored-resemblance: REGISTER_CONTENT is <537>\nline nebulous-booklet: REGISTER_CONTENT is <24135>\nline volatile-cuisine: REGISTER_CONTENT is <49233>\nline dizzy-latency: REGISTER_CONTENT is <41628>\nline vengeful-twig: REGISTER_CONTENT is <48961>\nline boundless-brocolli: REGISTER_CONTENT is <23356>\nline parched-hydrolyze: REGISTER_CONTENT is <42957>\nline muddled-township: REGISTER_CONTENT is <41035>\nline disagreeable-gradient: REGISTER_CONTENT is <28676>\nline jagged-valance: REGISTER_CONTENT is <46948>\nline capricious-shrine: REGISTER_CONTENT is <3593>\nline attractive-middleman: REGISTER_CONTENT is <13640>\nline festive-retrospectivity: REGISTER_CONTENT is <49557>\nline clumsy-harpsichord: REGISTER_CONTENT is <7165>\nline nutritious-wording: REGISTER_CONTENT is <22973>\nline cheerful-treasure: REGISTER_CONTENT is <22537>\nline loud-river: REGISTER_CONTENT is <38035>\nline womanly-retirement: REGISTER_CONTENT is <15219>\nline quick-pottery: REGISTER_CONTENT is <23287>\nline makeshift-ballpark: REGISTER_CONTENT is <5035>\nline obtainable-beanstalk: REGISTER_CONTENT is <21482>\nline illustrious-beat: REGISTER_CONTENT is <46227>\nline abaft-nursery: REGISTER_CONTENT is <28615>\nline plain-glance: REGISTER_CONTENT is <24799>\nline unsightly-bibliography: REGISTER_CONTENT is <42091>\nline kind-pear: REGISTER_CONTENT is <30820>\nline boiling-book: REGISTER_CONTENT is <9095>\nline erratic-jumper: REGISTER_CONTENT is <9835>\nline open-flood: REGISTER_CONTENT is <4326>\nline optimal-inability: REGISTER_CONTENT is <23493>\nline screeching-address: REGISTER_CONTENT is <34282>\nline shy-overload: REGISTER_CONTENT is <10145>\nline aback-mailer: REGISTER_CONTENT is <33518>\nline harmonious-oboe: REGISTER_CONTENT is <40748>\nline childlike-snob: REGISTER_CONTENT is <18525>\nline shivering-menorah: REGISTER_CONTENT is <20417>\nline callous-volatility: REGISTER_CONTENT is <3420>\nline parsimonious-ashtray: REGISTER_CONTENT is <46678>\nline lyrical-screw-up: REGISTER_CONTENT is <14388>\nline wide-laparoscope: REGISTER_CONTENT is <25881>\nline aboard-sermon: REGISTER_CONTENT is <22331>\nline juvenile-optimization: REGISTER_CONTENT is <38392>\nline quizzical-risk: REGISTER_CONTENT is <15452>\nline wooden-instrumentalist: REGISTER_CONTENT is <49462>\nline smelly-saffron: REGISTER_CONTENT is <15900>\nline aggressive-sultan: REGISTER_CONTENT is <15579>\nline soft-mom: REGISTER_CONTENT is <22512>\nline ablaze-molar: REGISTER_CONTENT is <33968>\nline bad-tectonics: REGISTER_CONTENT is <3211>\nline crabby-sister-in-law: REGISTER_CONTENT is <40818>\nline outstanding-foundation: REGISTER_CONTENT is <27529>\nline gifted-media: REGISTER_CONTENT is <32464>\nline earsplitting-hate: REGISTER_CONTENT is <46335>\nline wicked-jeans: REGISTER_CONTENT is <20461>\nline real-pollution: REGISTER_CONTENT is <9647>\nline majestic-tribe: REGISTER_CONTENT is <31690>\nline frantic-barrel: REGISTER_CONTENT is <787>\nline bashful-grandparent: REGISTER_CONTENT is <28491>\nline curious-clarity: REGISTER_CONTENT is <40734>\nline homeless-fitness: REGISTER_CONTENT is <33611>\nline super-dolor: REGISTER_CONTENT is <49140>\nline minor-lead: REGISTER_CONTENT is <32220>\nline ethereal-wholesaler: REGISTER_CONTENT is <48829>\nline productive-moccasins: REGISTER_CONTENT is <10773>\nline average-dibble: REGISTER_CONTENT is <30534>\nline exclusive-impostor: REGISTER_CONTENT is <19153>\nline ambitious-corps: REGISTER_CONTENT is <39649>\nline secretive-shack: REGISTER_CONTENT is <36987>\nline available-puritan: REGISTER_CONTENT is <15553>\nline tough-journey: REGISTER_CONTENT is <39095>\nline cuddly-cloak: REGISTER_CONTENT is <7023>\nline painstaking-gain: REGISTER_CONTENT is <8464>\nline shrill-forearm: REGISTER_CONTENT is <14824>\nline irate-native: REGISTER_CONTENT is <14776>\nline belligerent-afterthought: REGISTER_CONTENT is <47530>\nline combative-vinyl: REGISTER_CONTENT is <15779>\nline animated-shadowbox: REGISTER_CONTENT is <5103>\nline defiant-coevolution: REGISTER_CONTENT is <41227>\nline aberrant-pink: REGISTER_CONTENT is <34645>\nline misty-automaton: REGISTER_CONTENT is <41652>\nline angry-shoelace: REGISTER_CONTENT is <48798>\nline inconclusive-kennel: REGISTER_CONTENT is <13721>\nline gamy-atom: REGISTER_CONTENT is <25781>\nline coherent-banner: REGISTER_CONTENT is <14137>\nline abundant-doe: REGISTER_CONTENT is <19612>\nline horrible-colonization: REGISTER_CONTENT is <4424>\nline large-strobe: REGISTER_CONTENT is <25799>\nline brawny-bidet: REGISTER_CONTENT is <991>\nline succinct-overheard: REGISTER_CONTENT is <17902>\nline gruesome-architecture: REGISTER_CONTENT is <38852>\nline determined-trellis: REGISTER_CONTENT is <28961>\nline drab-leather: REGISTER_CONTENT is <19038>\nline lamentable-mutt: REGISTER_CONTENT is <1385>\nline verdant-circuit: REGISTER_CONTENT is <27517>\nline uttermost-counsel: REGISTER_CONTENT is <44716>\nline scarce-leeway: REGISTER_CONTENT is <8503>\nline wrong-bower: REGISTER_CONTENT is <23440>\nline evasive-urn: REGISTER_CONTENT is <298>\nline lopsided-insight: REGISTER_CONTENT is <3570>\nline rural-advance: REGISTER_CONTENT is <9060>\nline grouchy-creativity: REGISTER_CONTENT is <4891>\nline capable-formicarium: REGISTER_CONTENT is <28211>\nline lavish-oxygen: REGISTER_CONTENT is <43495>\nline heavenly-sensibility: REGISTER_CONTENT is <35706>\nline numerous-coordinator: REGISTER_CONTENT is <11919>\nline wiry-claim: REGISTER_CONTENT is <4615>\nline habitual-brake: REGISTER_CONTENT is <27011>\nline wealthy-deed: REGISTER_CONTENT is <37120>\nline eatable-credit: REGISTER_CONTENT is <3819>\nline amuck-explorer: REGISTER_CONTENT is <26550>\nline draconian-lumberman: REGISTER_CONTENT is <32581>\nline strong-replica: REGISTER_CONTENT is <20099>\nline lonely-winter: REGISTER_CONTENT is <38901>\nline addicted-copper: REGISTER_CONTENT is <40219>\nline innate-security: REGISTER_CONTENT is <33173>\nline racial-spiderling: REGISTER_CONTENT is <45521>\nline tacky-imitation: REGISTER_CONTENT is <23241>\nline abandoned-chit-chat: REGISTER_CONTENT is <33>\nline gaping-timetable: REGISTER_CONTENT is <7975>\nline tranquil-cop-out: REGISTER_CONTENT is <45375>\nline obsequious-bird-watcher: REGISTER_CONTENT is <29094>\nline condemned-teammate: REGISTER_CONTENT is <16158>\nline premium-slump: REGISTER_CONTENT is <45414>\nline many-widow: REGISTER_CONTENT is <13323>\nline sleepy-payee: REGISTER_CONTENT is <10087>\nline exultant-analogue: REGISTER_CONTENT is <6718>\nline imminent-mover: REGISTER_CONTENT is <36744>\nline unbecoming-delight: REGISTER_CONTENT is <2596>\nline chivalrous-adult: REGISTER_CONTENT is <26801>\nline recondite-satellite: REGISTER_CONTENT is <26125>\nline grubby-fertilizer: REGISTER_CONTENT is <34089>\nline momentous-concentrate: REGISTER_CONTENT is <45339>\nline fluffy-monopoly: REGISTER_CONTENT is <2064>\nline synonymous-pupil: REGISTER_CONTENT is <31315>\nline eager-convertible: REGISTER_CONTENT is <44515>\nline elated-ford: REGISTER_CONTENT is <12282>\nline knowing-agenda: REGISTER_CONTENT is <41042>\nline torpid-client: REGISTER_CONTENT is <13888>\nline exotic-rug: REGISTER_CONTENT is <30259>\nline testy-nestmate: REGISTER_CONTENT is <33213>\nline sharp-sculptural: REGISTER_CONTENT is <15111>\nline tightfisted-nutrition: REGISTER_CONTENT is <19168>\nline colossal-catcher: REGISTER_CONTENT is <781>\nline watery-tuxedo: REGISTER_CONTENT is <17815>\nline receptive-spending: REGISTER_CONTENT is <727>\nline flippant-bagpipe: REGISTER_CONTENT is <18079>\nline scientific-size: REGISTER_CONTENT is <7199>\nline ripe-hail: REGISTER_CONTENT is <11389>\nline mundane-picture: REGISTER_CONTENT is <34031>\nline dashing-mug: REGISTER_CONTENT is <5837>\nline disgusted-mapping: REGISTER_CONTENT is <47284>\nline abject-talking: REGISTER_CONTENT is <7095>\nline greasy-lyocell: REGISTER_CONTENT is <34642>\nline jittery-sarong: REGISTER_CONTENT is <10474>\nline alleged-lack: REGISTER_CONTENT is <7374>\nline clever-ambiguity: REGISTER_CONTENT is <31862>\nline deranged-woman: REGISTER_CONTENT is <24127>\nline stupid-fun: REGISTER_CONTENT is <7312>\nline filthy-prow: REGISTER_CONTENT is <34951>\nline illegal-pillar: REGISTER_CONTENT is <47797>\nline level-zoologist: REGISTER_CONTENT is <39771>\nline brainy-leash: REGISTER_CONTENT is <28568>\nline rare-stonework: REGISTER_CONTENT is <39779>\nline aboriginal-survivor: REGISTER_CONTENT is <30204>\nline puny-block: REGISTER_CONTENT is <14256>\nline envious-sprag: REGISTER_CONTENT is <11848>\nline needy-attendance: REGISTER_CONTENT is <9662>\nline boring-veil: REGISTER_CONTENT is <34284>\nline swift-universe: REGISTER_CONTENT is <13563>\nline whimsical-series: REGISTER_CONTENT is <13152>\nline dead-truth: REGISTER_CONTENT is <17616>\nline teeny-tofu: REGISTER_CONTENT is <17266>\nline lazy-den: REGISTER_CONTENT is <9711>\nline gusty-nibble: REGISTER_CONTENT is <6746>\nline zippy-sector: REGISTER_CONTENT is <23934>\nline new-distributor: REGISTER_CONTENT is <34884>\nline elderly-ingredient: REGISTER_CONTENT is <22452>\nline panoramic-plow: REGISTER_CONTENT is <45358>\nline lively-essential: REGISTER_CONTENT is <32204>\nline nauseating-democrat: REGISTER_CONTENT is <354>\nline wrathful-instructor: REGISTER_CONTENT is <49913>\nline fantastic-fear: REGISTER_CONTENT is <37986>\nline quiet-omega: REGISTER_CONTENT is <1931>\nline uninterested-dollar: REGISTER_CONTENT is <41484>\nline agonizing-drizzle: REGISTER_CONTENT is <10174>\nline therapeutic-castle: REGISTER_CONTENT is <18241>\nline flat-inside: REGISTER_CONTENT is <34156>\nline sour-checkroom: REGISTER_CONTENT is <19178>\nline glamorous-pillow: REGISTER_CONTENT is <25144>\nline ambiguous-glutamate: REGISTER_CONTENT is <44779>\nline happy-passage: REGISTER_CONTENT is <38426>\nline weary-coonskin: REGISTER_CONTENT is <1066>\nline naughty-tale: REGISTER_CONTENT is <11867>\nline knotty-extinction: REGISTER_CONTENT is <23846>\nline thinkable-proposition: REGISTER_CONTENT is <33234>\nline tasteful-vaulting: REGISTER_CONTENT is <36889>\nline uptight-tonality: REGISTER_CONTENT is <2587>\nline pastoral-eyrie: REGISTER_CONTENT is <40496>\n'
}

model_dir = "lmsys/vicuna-7b-v1.5"
res = to_multi_round_conversation(example, model_dir, 5)