import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from match import match
from typing import List, Callable, Tuple, Any, Match


def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    results = wikipedia.search(title)
    return WikipediaPage(results[0]).html()


def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all(class_="infobox")

    if not results:
        raise LookupError("Page has no infobox")
    return results[0].text


def clean_text(text: str) -> str:
    """Cleans given text removing non-ASCII characters and duplicate spaces & newlines

    Args:
        text - text to clean

    Returns:
        cleaned text
    """
    only_ascii = "".join([char if char in string.printable else " " for char in text])
    no_dup_spaces = re.sub(" +", " ", only_ascii)
    no_dup_newlines = re.sub("\n+", "\n", no_dup_spaces)
    return no_dup_newlines


def get_match(
    text: str,
    pattern: str,
    error_text: str = "Page doesn't appear to have the property you're expecting",
) -> Match:
    """Finds regex matches for a pattern

    Args:
        text - text to search within
        pattern - pattern to attempt to find within text
        error_text - text to display if pattern fails to match

    Returns:
        text that matches
    """
    p = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    match = p.search(text)

    if not match:
        raise AttributeError(error_text)
    return match


def get_polar_radius(planet_name: str) -> str:
    """Gets the radius of the given planet

    Args:
        planet_name - name of the planet to get radius of

    Returns:
        radius of the given planet
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(planet_name)))
    pattern = r"(?:Polar radius.*?)(?: ?[\d]+ )?(?P<radius>[\d,.]+)(?:.*?)km"
    error_text = "Page infobox has no polar radius information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("radius")


def get_birth_date(name: str) -> str:
    """Gets birth date of the given person

    Args:
        name - name of the person

    Returns:
        birth date of the given person
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"(?:Born\D*)(?P<birth>\d{4}-\d{2}-\d{2})"
    error_text = (
        "Page infobox has no birth information (at least none in xxxx-xx-xx format)"
    )
    match = get_match(infobox_text, pattern, error_text)

    return match.group("birth")

def get_population(location_name: str) -> str:
    """Gets the population of the given country or city

    Args:
        location_name - name of the country or city to get population of

    Returns:
        population of the given country or city
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(location_name)))
    pattern = r'<td class="infobox-data">(?:.|\n)*?"(?P<population>[\d,]+)"'
    error_text = "Page infobox has no population information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("population")

def get_official_language(country_name: str) -> str:
    """Gets the official language(s) of the given country

    Args:
        country_name - name of the country to get official language(s) of

    Returns:
        official language(s) of the given country
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(country_name)))
    pattern = r"(?:Official|National)\s*language(?:s)?\s*[:\-]?\s*(?P<languages>[A-Za-z\s,/-]+)"
    error_text = "Page infobox has no official language information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("languages")

def get_birth_place(person_name: str) -> str:
    """Gets the birthplace of the given person from Wikipedia infobox text

    Args:
        person_name - name of the person to get birthplace of

    Returns:
        birthplace of the given person
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(person_name)))

    pattern = (
        r"Born(?:\([^)]+\))?\s*[A-Za-z]+\s*\d{1,2},\s*\d{4}"           # Birth date
        r"(?:\s*\(age\s*\d+\))?"                                       # Optional (age ...)
        r"\s*(?P<birthplace>[A-Za-z\s.]+,\s*[A-Za-z\s.]+(?:,\s*[A-Za-z\s.]+)?)"  # Birthplace
    )

    error_text = "Page infobox has no birthplace information"
    match = get_match(infobox_text, pattern, error_text)

    # Optional: strip trailing country if only city and state are needed
    birthplace = match.group("birthplace").strip()

    # Optional normalization: keep only "City, State" if desired
    parts = [part.strip() for part in birthplace.split(',')]
    if len(parts) >= 2:
        birthplace = f"{parts[0]}, {parts[1]}"

    return birthplace




# below are a set of actions. Each takes a list argument and returns a list of answers
# according to the action and the argument. It is important that each function returns a
# list of the answer(s) and not just the answer itself.


def birth_date(matches: List[str]) -> List[str]:
    """Returns birth date of named person in matches

    Args:
        matches - match from pattern of person's name to find birth date of

    Returns:
        birth date of named person
    """
    return [get_birth_date(" ".join(matches))]


def polar_radius(matches: List[str]) -> List[str]:
    """Returns polar radius of planet in matches

    Args:
        matches - match from pattern of planet to find polar radius of

    Returns:
        polar radius of planet
    """
    return [get_polar_radius(matches[0])]

def population(matches: List[str]) -> List[str]:
    """Returns the population of a location in matches

    Args:
        matches - match from pattern of location to find population of

    Returns:
        population of the location
    """
    return [get_population(" ".join(matches))]

def official_language(matches: List[str]) -> List[str]:
    """Returns the official language(s) of the country in matches

    Args:
        matches - match from pattern of country to find official language(s) of

    Returns:
        official language(s) of the country
    """
    return [get_official_language(" ".join(matches))]

def birth_place(matches: List[str]) -> List[str]:
    """Returns birth place (city and country) of named person in matches

    Args:
        matches - match from pattern of person's name to find the birth place of

    Returns:
        Birth place of the named person
    """
    return [get_birth_place(" ".join(matches))]

# dummy argument is ignored and doesn't matter
def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt


# type aliases to make pa_list type more readable, could also have written:
# pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [...]
Pattern = List[str]
Action = Callable[[List[str]], List[Any]]

# The pattern-action list for the natural language query system. It must be declared
# here, after all of the function definitions
pa_list: List[Tuple[Pattern, Action]] = [
    ("when was % born".split(), birth_date),
    ("what is the polar radius of %".split(), polar_radius),
    ("what is the population of %".split(), population),
    ("what is the official language of %".split(), official_language),
    ("where was % born".split(), birth_place),
    (["bye"], bye_action),
]


def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    for pat, act in pa_list:
        mat = match(pat, src)
        if mat is not None:
            answer = act(mat)
            return answer if answer else ["No answers"]

    return ["I don't understand"]


def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully"""
    print("Welcome to the movie database!\n")
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nSo long!\n")


# uncomment the next line once you've implemented everything are ready to try it out
query_loop()
