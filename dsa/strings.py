"""
Strings: From Basics to Advanced Algorithms and Data Structures

This module provides a comprehensive overview of string operations, algorithms,
and data structures, ranging from basic manipulations to advanced techniques.
Each section is designed to enhance understanding and provide optimized, maintainable
implementations adhering to best practices.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import List, Dict, Optional
import torch


# ============================
# Basic String Operations
# ============================

def reverse_string(s: str) -> str:
    """
    Reverse a given string.

    Args:
        s (str): The string to reverse.

    Returns:
        str: The reversed string.
    """
    return s[::-1]


def is_palindrome(s: str) -> bool:
    """
    Check if the given string is a palindrome.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if palindrome, False otherwise.
    """
    return s == reverse_string(s)


def count_vowels(s: str) -> int:
    """
    Count the number of vowels in a string.

    Args:
        s (str): The string in which to count vowels.

    Returns:
        int: The count of vowels.
    """
    vowels = set('aeiouAEIOU')
    return sum(1 for char in s if char in vowels)


def to_uppercase(s: str) -> str:
    """
    Convert all characters in the string to uppercase.

    Args:
        s (str): The input string.

    Returns:
        str: The uppercase string.
    """
    return s.upper()


def to_lowercase(s: str) -> str:
    """
    Convert all characters in the string to lowercase.

    Args:
        s (str): The input string.

    Returns:
        str: The lowercase string.
    """
    return s.lower()


# ============================
# Advanced String Algorithms
# ============================

def compute_lps_array(pattern: str) -> List[int]:
    """
    Compute the Longest Prefix Suffix (LPS) array used in KMP algorithm.

    Args:
        pattern (str): The pattern string.

    Returns:
        List[int]: The LPS array.
    """
    lps = [0] * len(pattern)
    length = 0  # length of the previous longest prefix suffix

    # The loop calculates lps[i] for i from 1 to len(pattern)-1
    for i in range(1, len(pattern)):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]

        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
        else:
            lps[i] = 0

    return lps


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Knuth-Morris-Pratt (KMP) algorithm for substring search.

    Args:
        text (str): The text to search within.
        pattern (str): The pattern to search for.

    Returns:
        List[int]: List of starting indices where pattern is found in text.
    """
    if not pattern:
        raise ValueError("Pattern must be non-empty")

    lps = compute_lps_array(pattern)
    result = []
    i = j = 0  # index for text, index for pattern

    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1

            if j == len(pattern):
                result.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return result


def rabin_karp_search(text: str, pattern: str, prime: int = 101) -> List[int]:
    """
    Rabin-Karp algorithm for substring search using hashing.

    Args:
        text (str): The text to search within.
        pattern (str): The pattern to search for.
        prime (int): A prime number for hashing.

    Returns:
        List[int]: List of starting indices where pattern is found in text.
    """
    n = len(text)
    m = len(pattern)
    if m > n:
        return []

    pattern_hash = 0
    text_hash = 0
    h = 1  # value of h = pow(d, m-1) % prime
    d = 256  # number of characters in the input alphabet

    for _ in range(m - 1):
        h = (h * d) % prime

    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime

    result = []
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i + m] == pattern:
                result.append(i)

        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime

    return result


def longest_common_subsequence(s1: str, s2: str) -> str:
    """
    Compute the Longest Common Subsequence (LCS) between two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        str: The LCS string.
    """
    m, n = len(s1), len(s2)
    dp = [["" for _ in range(n + 1)] for __ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + s1[i]
            else:
                dp[i + 1][j + 1] = dp[i][j + 1] if len(dp[i][j + 1]) > len(dp[i + 1][j]) else dp[i + 1][j]

    return dp[m][n]


def edit_distance(s1: str, s2: str) -> int:
    """
    Compute the edit distance (Levenshtein distance) between two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The edit distance.
    """
    m, n = len(s1), len(s2)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[m][n].item()


# ============================
# String Data Structures
# ============================

class TrieNode:
    """
    A node in the Trie structure.
    """

    def __init__(self) -> None:
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False


class Trie:
    """
    Trie data structure for efficient word storage and retrieval.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Insert a word into the Trie.

        Args:
            word (str): The word to insert.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Search for a word in the Trie.

        Args:
            word (str): The word to search for.

        Returns:
            bool: True if the word exists, False otherwise.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        Check if there is any word in the Trie that starts with the given prefix.

        Args:
            prefix (str): The prefix to check.

        Returns:
            bool: True if such a prefix exists, False otherwise.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class SuffixTrieNode:
    """
    A node in the Suffix Trie structure.
    """

    def __init__(self) -> None:
        self.children: Dict[str, 'SuffixTrieNode'] = {}
        self.indices: List[int] = []


class SuffixTrie:
    """
    Suffix Trie data structure for efficient substring search.
    """

    def __init__(self, text: str) -> None:
        self.root = SuffixTrieNode()
        self.text = text
        self.build_suffix_trie()

    def build_suffix_trie(self) -> None:
        """
        Build the suffix trie for the given text.
        """
        for i in range(len(self.text)):
            current_node = self.root
            for char in self.text[i:]:
                if char not in current_node.children:
                    current_node.children[char] = SuffixTrieNode()
                current_node = current_node.children[char]
                current_node.indices.append(i)

    def search(self, pattern: str) -> List[int]:
        """
        Search for all occurrences of the pattern in the text.

        Args:
            pattern (str): The pattern to search for.

        Returns:
            List[int]: List of starting indices where pattern is found.
        """
        current_node = self.root
        for char in pattern:
            if char not in current_node.children:
                return []
            current_node = current_node.children[char]
        return current_node.indices


class SuffixAutomatonNode:
    """
    A node in the Suffix Automaton.
    """

    def __init__(self) -> None:
        self.next: Dict[str, 'SuffixAutomatonNode'] = {}
        self.link: Optional['SuffixAutomatonNode'] = None
        self.len: int = 0


class SuffixAutomaton:
    """
    Suffix Automaton data structure for efficient substring operations.
    """

    def __init__(self) -> None:
        self.size = 1
        self.last = self._create_node(0)
        self.nodes: List[SuffixAutomatonNode] = [self.last]

    def _create_node(self, length: int) -> SuffixAutomatonNode:
        node = SuffixAutomatonNode()
        node.len = length
        return node

    def add_char(self, char: str) -> None:
        """
        Add a character to the automaton.

        Args:
            char (str): The character to add.
        """
        current = self._create_node(self.nodes[self.last.len].len + 1)
        self.nodes.append(current)
        self.size += 1
        p = self.last

        while p and char not in p.next:
            p.next[char] = current
            p = p.link

        if not p:
            current.link = self.nodes[0]
        else:
            q = p.next[char]
            if p.len + 1 == q.len:
                current.link = q
            else:
                clone = self._create_node(p.len + 1)
                clone.next = q.next.copy()
                clone.link = q.link
                self.nodes.append(clone)
                self.size += 1
                while p and p.next[char] == q:
                    p.next[char] = clone
                    p = p.link
                q.link = clone
                current.link = clone

        self.last = current

    def build_automaton(self, s: str) -> None:
        """
        Build the suffix automaton for the given string.

        Args:
            s (str): The string to build the automaton for.
        """
        for char in s:
            self.add_char(char)

    def contains(self, substring: str) -> bool:
        """
        Check if the automaton contains the given substring.

        Args:
            substring (str): The substring to check.

        Returns:
            bool: True if substring exists, False otherwise.
        """
        current = self.nodes[0]
        for char in substring:
            if char not in current.next:
                return False
            current = current.next[char]
        return True


# ============================
# Utility Functions
# ============================

def find_all_unique_substrings(s: str) -> int:
    """
    Find the number of unique substrings in the given string using Suffix Automaton.

    Args:
        s (str): The input string.

    Returns:
        int: The number of unique substrings.
    """
    automaton = SuffixAutomaton()
    automaton.build_automaton(s)
    count = 0
    for node in automaton.nodes[1:]:
        count += node.len - node.link.len
    return count


# ============================
# Example Usage and Test Cases
# ============================

def main() -> None:
    """
    Main function to demonstrate the usage of string algorithms and data structures.
    """
    sample_text = "abracadabra"
    pattern = "abra"

    # Basic Operations
    print(f"Original String: {sample_text}")
    reversed_str = reverse_string(sample_text)
    print(f"Reversed String: {reversed_str}")
    print(f"Is Palindrome: {is_palindrome(sample_text)}")
    print(f"Vowel Count: {count_vowels(sample_text)}")
    print(f"Uppercase: {to_uppercase(sample_text)}")
    print(f"Lowercase: {to_lowercase(sample_text)}\n")

    # KMP Search
    kmp_indices = kmp_search(sample_text, pattern)
    print(f"KMP Search for '{pattern}' in '{sample_text}': {kmp_indices}")

    # Rabin-Karp Search
    rk_indices = rabin_karp_search(sample_text, pattern)
    print(f"Rabin-Karp Search for '{pattern}' in '{sample_text}': {rk_indices}\n")

    # Longest Common Subsequence
    s1 = "abcdef"
    s2 = "acbcf"
    lcs = longest_common_subsequence(s1, s2)
    print(f"LCS of '{s1}' and '{s2}': {lcs}")

    # Edit Distance
    ed = edit_distance(s1, s2)
    print(f"Edit Distance between '{s1}' and '{s2}': {ed}\n")

    # Trie Usage
    trie = Trie()
    words = ["apple", "app", "apply", "apt", "bat", "batch"]
    for word in words:
        trie.insert(word)

    search_word = "app"
    print(f"Trie Search for '{search_word}': {trie.search(search_word)}")
    prefix = "ap"
    print(f"Trie starts with '{prefix}': {trie.starts_with(prefix)}\n")

    # Suffix Trie Usage
    suffix_trie = SuffixTrie(sample_text)
    substring = "rac"
    suffix_indices = suffix_trie.search(substring)
    print(f"Suffix Trie Search for '{substring}' in '{sample_text}': {suffix_indices}\n")

    # Suffix Automaton Usage
    suffix_automaton = SuffixAutomaton()
    suffix_automaton.build_automaton(sample_text)
    check_substring = "cad"
    contains = suffix_automaton.contains(check_substring)
    print(f"Suffix Automaton contains '{check_substring}': {contains}")

    # Unique Substrings
    unique_subs = find_all_unique_substrings(sample_text)
    print(f"Number of unique substrings in '{sample_text}': {unique_subs}")


if __name__ == "__main__":
    main()