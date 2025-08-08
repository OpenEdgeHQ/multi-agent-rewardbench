import re
import trafilatura

def clean_html(html):
    return trafilatura.html2txt(html, clean=True)

# def clean_html(txt):
#     """ Clean HTML tags of webpages downloaded
#     Use this function for Gutenberg book format.
#     """
#     style_tag_re = re.compile('<style.*?>[^<>]*?</style>')
#     txt = re.sub(style_tag_re, ' ', txt)
#     script_tag_re = re.compile('<script.*?>[^<>]*?</script>')
#     txt = re.sub(script_tag_re, ' ', txt)
#     doc_tag_re = re.compile('<!DOCTYPE[^<>]*?>')
#     txt = re.sub(doc_tag_re, ' ', txt)
#     html_tag_re = re.compile('<.*?>')
#     txt = connect_lines(txt)
#     return re.sub(html_tag_re, ' ', txt).strip()
#
#
# def connect_lines(txt, line_sep='\n'):
#     """ This happens when you crawl text from a webpage and
#     they have random breaking lines mid-sentence.
#
#     This function is to connect those lines.
#
#     Two consecutive lines are separated by line_sep.
#     """
#     lines = txt.split('\n')
#
#     result, curr = '', ''
#     for line in lines:
#         line = line.strip()
#         if not line:
#             if curr:
#                 result += (curr + '\n')
#             result += line_sep
#             curr = ''
#         else:
#             curr += (line + ' ')
#
#     return result + curr