# Author: Zhiqiu Lin
import os
import json

import torch
from torchvision.datasets.folder import default_loader


from collections import OrderedDict
import warnings

DATASET_PATHS = {  # Modify if you saved these datasets elsewhere
    # 'imagenet': '/scratch/',
    'imagenet': '/scratch/jiashi/',
    'imagenetv2': "/scratch/jiashi/",
    'imagenet_sketch': "/scratch/jiashi/",
    'imagenet_a': "/scratch/jiashi/",
    'imagenet_r': "/scratch/jiashi/",
    'objectnet': "/scratch/jiashi/",
}

def get_lab2cname(data_source):
    """Get a label-to-classname mapping (dict).

    Args:
        data_source (list): a list of dict.
    """
    container = set()
    for item in data_source:
        container.add((item['label'], item['classname']))
    mapping = {label: classname for label, classname in container}
    labels = list(mapping.keys())
    labels.sort()
    classnames = [mapping[label] for label in labels]
    return mapping, classnames


def get_num_classes(data_source):
    """Count number of classes.

    Args:
        data_source (list): a list of Datum objects.
    """
    label_set = set()
    for item in data_source:
        label_set.add(item['label'])
    return max(label_set) + 1


class Benchmark(object):
    """A benchmark that contains 
    1) training data
    2) validation data
    3) test data
    """

    dataset_name = "" # e.g. imagenet, etc.

    def __init__(self, train=None, val=None, test=None):
        self.train = train  # labeled training data source
        self.val = val  # validation data source
        self.test = test  # test data source
        self.num_classes = get_num_classes(train)
        self.lab2cname, self.classnames = get_lab2cname(train)


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = os.path.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def load_json(json_location, default_obj=None):
    '''Load a json file.'''
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj


def read_split(filepath, path_prefix):
    '''Read train/val/test split from a json file.'''
    def _convert(items):
        '''Convert a list of items to a list of dict.'''
        lst = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            check_isfile(impath)
            item = {'impath': impath,
                    'label': int(label),
                    'classname': classname}
            lst.append(item)
        return lst

    print(f"Reading split from {filepath}")
    split = load_json(filepath)
    train = _convert(split["train"])
    val = _convert(split["val"])
    test = _convert(split["test"])

    return train, val, test

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames


class ImageNet(Benchmark):

    dataset_name = "imagenet"
    split_google_url = "https://drive.google.com/file/d/1SvPIN6iV6NP2Oulj19a869rBXrB5SNFo/view"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_ImageNet.json")

        if not os.path.exists(self.split_path):
            print(
                f"Please download the split path from {self.split_google_url}"
                f" and put it to {self.split_path}")
            raise FileNotFoundError(self.split_path)
        train=torch.load('/scratch/jiashi/train_imagenet_list')
        val=torch.load('/scratch/jiashi/val_imagenet_list')
        test=torch.load('/scratch/jiashi/test_imagenet_list')
        # train, val, test = read_split(self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = {'impath': impath,
                        'label': label,
                        'classname': classname}
                items.append(item)

        return items
    

TO_BE_IGNORED = ["README.txt"]


class ImageNetA(Benchmark):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_name = "imagenet-adversarial"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.original_imagenet_dir = os.path.join(root, "imagenet")
        original_text_file = os.path.join(self.original_imagenet_dir, "classnames.txt")
        original_classnames = read_classnames(original_text_file)

        self.image_dir = os.path.join(self.dataset_dir, "imagenet-a")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data, label_map = self.read_data(classnames, original_classnames)
        self.label_map = label_map
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames, original_classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]

        original_folders = [folder for folder in original_classnames]
        label_map = [original_folders.index(folder) for folder in folders]

        items = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = {"impath": impath, "label": label, "classname": classname}
                items.append(item)

        return items, label_map


class ImageNetR(Benchmark):
    """ImageNet-R(endition).

    This dataset is used for testing only.
    """

    dataset_name = "imagenet-rendition"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.original_imagenet_dir = os.path.join(root, "imagenet")
        original_text_file = os.path.join(self.original_imagenet_dir, "classnames.txt")
        original_classnames = read_classnames(original_text_file)

        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data, label_map = self.read_data(classnames, original_classnames)
        self.label_map = label_map

        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames, original_classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]

        original_folders = [folder for folder in original_classnames]
        label_map = [original_folders.index(folder) for folder in folders]

        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = {"impath": impath, "label": label, "classname": classname}
                items.append(item)

        return items, label_map
    
    
class ImageNetSketch(Benchmark):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_name = "imagenet-sketch"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = {"impath": impath, "label": label, "classname": classname}
                items.append(item)

        return items


class ImageNetV2(Benchmark):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_name = "imagenetv2"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = {"impath": impath, "label": label, "classname": classname}
                items.append(item)

        return items



class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform, root=None):
        self.data_source = data_source
        self.transform = transform
        if root is not None:
            self.root = root
            for item in self.data_source:
                item['impath'] = os.path.join(self.root, item['impath'])

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        img = self.transform(default_loader(item['impath']))

        # output = {
        #     "img": img,
        #     "label": item['label'],
        #     "classname": item['classname'],
        #     "impath": item['impath'],
        # }

        return img, item['label']


openai_imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "eft", 
                           "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", 
                           "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", 
                           "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", 
                           "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox","maillot", 
                           "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", 
                           "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

# COOP 7 templates
IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

def build_few_shot_imagenet_dataset(train_shot,
                                    seed,
                                    preprocess,
                                    indices_dir,
                                    root_dir):
    few_shot_index_file = os.path.join(
        indices_dir, 'imagenet', f"shot_{train_shot}-seed_{seed}.json")
    assert os.path.exists(few_shot_index_file), f"Few-shot data does not exist at {few_shot_index_file}."
    few_shot_dataset = load_json(few_shot_index_file)
    return {
        'train': DatasetWrapper(few_shot_dataset['train']['data'], transform=preprocess, root=root_dir), # Contains train_shot train images
        'val': DatasetWrapper(few_shot_dataset['val']['data'], transform=preprocess, root=root_dir), # Contains min(4, train_shot)-shot val images
    }

def build_imagenet_dataset(dataset_name, split, preprocess):
    assert dataset_name in DATASET_PATHS
    
    dataset = imagenet_DATASETS[dataset_name](DATASET_PATHS[dataset_name])
    if dataset_name in ['imagenet_a', 'imagenet_r']:
        label_map = dataset.label_map
    else:
        label_map = list(range(1000))
    
    if split in ['train', 'val']:
        assert dataset_name in ['imagenet']
    
    split_set = getattr(dataset, split)
    testset = DatasetWrapper(split_set, transform=preprocess)
    return testset, openai_imagenet_classes, label_map

imagenet_DATASETS = {
    "imagenet": ImageNet,
    "imagenetv2": ImageNetV2,
    "imagenet_sketch": ImageNetSketch,
    "imagenet_a": ImageNetA,
    "imagenet_r": ImageNetR,
}

if __name__ == '__main__':
    for dataset_name in imagenet_DATASETS:
        testset, labels, label_map = build_imagenet_dataset(dataset_name, 'test', None)
        print(f"Dataset (test-split): {dataset_name}, Number of classes: {len(label_map)}, Number of images: {len(testset)}")
        
    print("I also provided train and val (8:2) split for ImageNet train set.")
    for split in ['train', 'val']:
        dataset, _, _ = build_imagenet_dataset('imagenet', split, None)
        print(f"Dataset ({split}-split): ImageNet, Number of classes: 1000, Number of images: {len(dataset)}")
        
    # build few shot dataset
    print(f"You can also get 1/2/4/8/16 train-shot dataset from build_few_shot_imagenet_dataset()")
    train_shot = 16 # you can use 1/2/4/8/16
    seed = 1 # you can use 1/2/3/4/5
    few_shot_dataset = build_few_shot_imagenet_dataset(train_shot, seed, None)
    print(f"Dataset (train-shot-{train_shot}): ImageNet, Number of train images: {len(few_shot_dataset['train'])}")
    print(f"Dataset (val-shot-{min(train_shot, 4)}): ImageNet, Number of val images: {len(few_shot_dataset['val'])}")