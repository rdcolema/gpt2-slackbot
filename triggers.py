import re


match_triggers = ()

search_triggers = (
    (re.compile("trump"), "trump"),
    (re.compile("okcupid"), "okcupid"),
    (re.compile("ruby"), "ruby"),
    (re.compile("rap"), "rap")
)


def get_response_key(command, regex_type='match'):
    regex = re.match if regex_type == 'match' else re.search
    lookup = match_triggers if regex_type == 'match' else search_triggers
    for key, value in lookup:
        if regex(key, command):
            return value
    return None
