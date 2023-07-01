import os
from typing import List, Dict
import re
import markdown2
import spacy


def get_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
    """
    Retrieve the paths of files in the given directory that match the specified file extensions.

    Args:
        directory (str): The directory path to search for files.
        extensions (list): A list of file extensions to match.

    Returns:
        list: A list of file paths that match the given extensions.
    """
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1]
            if file_extension in extensions:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    return file_paths


def extract_code_blocks_from_markdown(file_path: str) -> List[str]:
    """
    Extract code blocks from a Markdown file.

    Args:
        file_path (str): The path of the Markdown file.

    Returns:
        List[str]: A list of code blocks extracted from the Markdown file.
    """
    if not file_path.endswith(".md"):
        raise ValueError("The provided file is not a Markdown file.")

    code_blocks = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        code_block = []
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                if in_code_block:
                    code_blocks.append("".join(code_block))
                    code_block = []
                    in_code_block = False
                else:
                    in_code_block = True
            elif in_code_block:
                code_block.append(line)

    return code_blocks


def extract_links_from_markdown(file_path: str) -> List[str]:
    """
    Extract links from a Markdown file.

    Args:
        file_path (str): The path of the Markdown file.

    Returns:
        List[str]: A list of links extracted from the Markdown file.
    """
    if not file_path.endswith(".md"):
        raise ValueError("The provided file is not a Markdown file.")

    links = []

    with open(file_path, "r") as file:
        content = file.read()
        link_pattern = r"\[(.*?)\]\((.*?)\)"
        matches = re.findall(link_pattern, content)

        for match in matches:
            link_text, link_url = match
            links.append(link_url)

    return links


def extract_images_from_markdown(file_path: str) -> List[str]:
    """
    Extract image URLs from a Markdown file.

    Args:
        file_path (str): The path of the Markdown file.

    Returns:
        List[str]: A list of image URLs extracted from the Markdown file.
    """
    if not file_path.endswith(".md"):
        raise ValueError("The provided file is not a Markdown file.")

    images = []

    with open(file_path, "r") as file:
        content = file.read()
        image_pattern = r"!\[(.*?)\]\((.*?)\)"
        matches = re.findall(image_pattern, content)

        for match in matches:
            alt_text, image_url = match
            images.append(image_url)

    return images


def extract_headings_with_paragraphs_from_markdown(file_path: str) -> dict:
    """
    Extract headings and the paragraph text below each heading from a Markdown file.

    Args:
        file_path (str): The path of the Markdown file.

    Returns:
        dict: A dictionary where the keys are the headings and the values are the corresponding paragraphs.
    """
    if not file_path.endswith(".md"):
        raise ValueError("The provided file is not a Markdown file.")

    heading_paragraphs = {}

    with open(file_path, "r") as file:
        content = file.read()
        heading_pattern = r"#+\s(.+)"
        matches = re.findall(heading_pattern, content)

        for match in matches:
            heading = match
            next_line_index = content.index(match) + len(match) + 1
            next_line = content[next_line_index:].strip()

            if next_line.startswith("#"):
                paragraph = ""
            else:
                paragraph = next_line

            heading_paragraphs[heading] = paragraph

    return heading_paragraphs


def extract_tables_from_markdown(file_path: str) -> List[List[str]]:
    """
    Extract tables from a Markdown file.

    Args:
        file_path (str): The path of the Markdown file.

    Returns:
        List[List[str]]: A list of tables extracted from the Markdown file.
    """
    if not file_path.endswith(".md"):
        raise ValueError("The provided file is not a Markdown file.")

    tables = []

    with open(file_path, "r") as file:
        content = file.read()
        table_pattern = r"\|(.+)\|(\n\|.+)+\n?"
        matches = re.findall(table_pattern, content)

        for match in matches:
            table_lines = match[0].split("\n|")[1:]
            table = [line.strip().split("|") for line in table_lines]
            tables.append(table)

    return tables


def extract_project_description_from_readme(file_path: str) -> str:
    """
    Extract the project description from a README.md file.

    Args:
        file_path (str): The path of the README.md file.

    Returns:
        str: The project description extracted from the README.md file.
    """
    if not file_path.endswith(".md"):
        raise ValueError("The provided file is not a .md file.")

    with open(file_path, "r") as file:
        lines = file.readlines()
        description = ""
        in_description = False

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if not in_description:
                if line.lower().startswith("#"):
                    in_description = True
                    description += line.lstrip("#").strip() + " "
            else:
                if line.lower().startswith("#"):
                    break
                else:
                    description += line + " "

    return description.strip()


def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Convert Markdown text to html.

    Args:
        markdown_text (str): The Markdown text to be converted.

    Returns:
        str: The html equivalent text of the Markdown text.
    """
    plain_text = markdown2.markdown(
        markdown_text, extras=["tables", "fenced-code-blocks"])
    return plain_text


def convert_markdown_file_to_html(file_path: str) -> str:
    """
    Convert Markdown file to html.

    Args:
        file_path (str): The path to the Markdown file.

    Returns:
        str: The html equivalent content of the Markdown file.

    Raises:
        ValueError: If the file is not a Markdown file.
    """
    if not file_path.lower().endswith('.md'):
        raise ValueError("The file is not a Markdown file.")

    with open(file_path, 'r') as file:
        markdown_text = file.read()

    html_content = convert_markdown_to_html(markdown_text)

    return html_content


def check_phrase_similarity_using_spacyweb(phrase1: str, phrase2: str, threshold: float = 0.5) -> bool:
    """
    Checks the similarity between two phrases using spaCy's pre-trained word vectors.

    Args:
        phrase1 (str): The first phrase.
        phrase2 (str): The second phrase.
        threshold (float): The threshold similarity score.

    Returns:
        bool: True if the similarity score is above the threshold, False otherwise.
    """
    # python -m spacy download en_core_web_lg
    nlp = spacy.load("en_core_web_lg")

    doc1 = nlp(phrase1)
    doc2 = nlp(phrase2)

    similarity_score = doc1.similarity(doc2)

    return similarity_score >= threshold


def check_similarity(text1: str, text2: str, strategy: str = "in") -> bool:
    """
    Checks the similarity between two texts using different strategies.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.
        strategy (str, optional): The strategy to use for similarity check.
            Valid options are:
            - "in": Checks if one text is contained within the other.
            - "spacy_web": Checks similarity using spaCy's pre-trained word vectors.

    Returns:
        bool: True if the texts are similar based on the chosen strategy, False otherwise.
    """
    if strategy == "in":
        return (text1 in text2) or (text2 in text1)
    elif strategy == "spacy_web":
        return check_phrase_similarity_using_spacyweb(text1, text2, 0.5)


def remove_sections_from_markdown(markdown_content: List[str], headings: List[str], strategy: str = "in") -> List[str]:
    """
    Removes sections from Markdown content based on a heading and similarity strategy.

    Args:
        markdown_content (List[str]): The list of lines in the Markdown content.
        headings (List[str]): List of headings to search for and remove along with its sections.
        strategy (str, optional): The strategy to use for similarity check. Valid options are:
            - "in": Checks if the heading is contained within the line.
            - "spacy_web": Checks similarity using spaCy's pre-trained word vectors.

    Returns:
        List[str]: The updated Markdown content with the specified sections removed.
    """
    updated_content = []
    skip_section = False

    for line in markdown_content:
        for heading in headings:
            if check_similarity(heading, line, strategy):
                skip_section = True
                break
        if not skip_section and line.startswith('#'):
            skip_section = False

        if not skip_section:
            updated_content.append(line)

    return updated_content


def remove_headings_from_markdown_file(file_path: str, heading: str) -> List[str]:
    """
    Removes the specified heading and all the subsequent subheadings and paragraphs from the markdown file.

    Args:
        file_path (str): The path to the markdown file.
        heading (str): The heading to be removed along with its subsequent sections.

    Returns:
        List[str]: The updated markdown content with the specified heading and its subsequent sections removed.

    Raises:
        ValueError: If the file is not a Markdown file.
    """
    if not file_path.lower().endswith('.md'):
        raise ValueError("The file is not a Markdown file.")

    with open(file_path, 'r') as file:
        markdown_content = file.readlines()

    updated_content = remove_sections_from_markdown(markdown_content, heading)

    return updated_content


def get_elements_from_markdown_file(file_path: str, elements: List[str]) -> Dict[str, str]:
    """
    Extracts specific elements from a Markdown file.

    Args:
        file_path (str): The path to the Markdown file.
        elements (List[str]): A list of elements to extract. Valid options are:
            - "links": Extracts links from the Markdown file.
            - "images": Extracts images from the Markdown file.
            - "headings": Extracts headings with their corresponding paragraphs from the Markdown file.
            - "code": Extracts code blocks from the Markdown file.
            - "tables": Extracts tables from the Markdown file.
            - "description": Extracts the project description from a README file.

    Returns:
        Dict[str, str]: A dictionary containing the extracted elements as key-value pairs.
            The keys correspond to the requested elements, and the values contain the extracted content.

    Raises:
        ValueError: If the file is not a Markdown file.
    """
    if not file_path.lower().endswith('.md'):
        raise ValueError("The file is not a Markdown file.")

    elements_to_extract = {
        "links": extract_links_from_markdown,
        "images": extract_images_from_markdown,
        "headings": extract_headings_with_paragraphs_from_markdown,
        "code": extract_code_blocks_from_markdown,
        "tables": extract_tables_from_markdown,
        "description": extract_project_description_from_readme
    }

    result = {}

    for element in elements:
        if element not in elements_to_extract.keys():
            continue
        result[element] = elements_to_extract.get(element)(file_path)

    return result


def remove_images_from_markdown(file_path: str) -> str:
    """
    Removes image tags from a Markdown file and returns the updated content without images.

    Args:
        file_path: The path to the Markdown file.

    Returns:
        The Markdown content without images.

    Raises:
        ValueError: If the provided file is not a Markdown file or if the file does not exist.
    """

    if not file_path.lower().endswith('.md'):
        raise ValueError(
            "Invalid file. Only Markdown files (.md) are supported.")

    if not os.path.isfile(file_path):
        raise ValueError("File not found.")

    with open(file_path, 'r') as f:
        markdown_content = f.read()

    markdown_content_without_images = re.sub(
        '!\[.*?\]\(.*?\)', '', markdown_content)

    return markdown_content_without_images


def remove_links_from_markdown(file_path: str) -> str:
    """
    Removes link tags from a Markdown file and returns the updated content.

    Args:
        file_path: The path to the Markdown file.

    Returns:
        The Markdown content without links.

    Raises:
        ValueError: If the provided file is not a Markdown file or if the file does not exist.
    """

    if not file_path.lower().endswith('.md'):
        raise ValueError(
            "Invalid file. Only Markdown files (.md) are supported.")

    if not os.path.isfile(file_path):
        raise ValueError("File not found.")

    with open(file_path, 'r') as f:
        markdown_content = f.read()

    markdown_content_without_links = re.sub(
        '\[.*?\]\(.*?\)', '', markdown_content)

    return markdown_content_without_links


def remove_code_blocks_from_markdown(file_path: str) -> str:
    """
    Removes code blocks from a Markdown file and returns the updated content.

    Args:
        file_path: The path to the Markdown file.

    Returns:
        The Markdown content without code blocks.

    Raises:
        ValueError: If the provided file is not a Markdown file or if the file does not exist.
    """

    if not file_path.lower().endswith('.md'):
        raise ValueError(
            "Invalid file. Only Markdown files (.md) are supported.")

    if not os.path.isfile(file_path):
        raise ValueError("File not found.")

    with open(file_path, 'r') as f:
        markdown_content = f.read()

    markdown_content_without_code_blocks = re.sub(
        '```[\s\S]*?```', '', markdown_content)

    return markdown_content_without_code_blocks


def remove_tables_from_markdown(file_path: str) -> str:
    """
    Removes tables from a Markdown file and returns the updated content.

    Args:
        file_path: The path to the Markdown file.

    Returns:
        The Markdown content without tables.

    Raises:
        ValueError: If the provided file is not a Markdown file or if the file does not exist.
    """

    if not file_path.lower().endswith('.md'):
        raise ValueError(
            "Invalid file. Only Markdown files (.md) are supported.")

    if not os.path.isfile(file_path):
        raise ValueError("File not found.")

    with open(file_path, 'r') as f:
        markdown_content = f.read()

    markdown_content_without_tables = re.sub(
        r'\n\|.*\|\n\|.*\|\n(\|.*\|)+', '', markdown_content)

    return markdown_content_without_tables
