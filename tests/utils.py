from __future__ import annotations

import regex as re
import os, json
from collections import Counter
from .common import gpt2_bytes_to_unicode
from collections import defaultdict
import multiprocessing
from jaxtyping import Bool
from typing import BinaryIO


class token_node:
    __slots__ = ("val", "prev", "next", "freq", "alive")

    def __init__(self, val: str, freq : int):
        self.val : str = val
        self.prev : token_node | None = None
        self.next : token_node | None = None
        self.freq : int = freq
        self.alive : Bool = True

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    
    @staticmethod
    def encode_worker(
        texts: str, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> list[int]:
        
        bytes_to_token_id_map = {v : k for k, v in vocab.items()}
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        chunks = re.split("|".join(map(re.escape, special_tokens)), texts) if special_tokens else [texts]
        words: list[list[re.Match[str]]] = [re.finditer(pattern, chunk) for chunk in chunks]
        special_token_list : list[bytes] | None = [b"<|endoftext|>"] * (len(chunks) - 1) if special_tokens else None
        
        words_in_bytes : list[list[list[bytes]]] = []
        for chunk in words:
            chunk_in_bytes : list[list[bytes]] = []
            for word in chunk:
                bytestring = word.group().encode("utf-8")
                if not bytestring: continue
        
                chunk_in_bytes.append([b.to_bytes() for b in bytestring])
            
            words_in_bytes.append(chunk_in_bytes)
            
        cache : dict[bytes, list[int]] = defaultdict(list)
        token_ids : list[int] = []
        
        for chunk_in_bytes in words_in_bytes:
            for byte_list in chunk_in_bytes:
                
                bytestring = b"".join(byte_list)
                get_from_cache = cache.get(bytestring)
                if get_from_cache:
                    token_ids.extend(get_from_cache)
                    continue
                    
                for b0, b1 in merges:
                    if len(byte_list) == 1: break
                    
                    new_byte_list : list[bytes] = []
                    i = 0
                    while i < len(byte_list):
                        if i < len(byte_list) - 1 and byte_list[i] == b0 and byte_list[i + 1] == b1:
                            new_byte_list.append(b0 + b1)
                            i += 2
                        else:
                            new_byte_list.append(byte_list[i])
                            i += 1
                        
                    byte_list = new_byte_list    
                
                cache[bytestring] = [bytes_to_token_id_map[bytes] for bytes in byte_list]
                token_ids.extend(cache[bytestring])
                
            if special_token_list and len(special_token_list) > 0:
                token_ids.append(bytes_to_token_id_map[special_token_list.pop()])
                            
        return token_ids
        
    
    def encode(self, texts : str) -> list[int]:
        num_workers = 1
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(Tokenizer.encode_worker, [(texts, self.vocab, self.merges, self.special_tokens)])
            
        token_ids : list[int] = []
        for result in results: token_ids.extend(result)
        
        return token_ids
    
    
    @staticmethod
    def decode_worker(
        encoded_list: list[int], 
        vocab: dict[int, bytes],
    ) -> str:
        
        decoded_string = ""
        
        for token_id in encoded_list:
            bytestring = vocab[token_id]
            unicode_string = bytestring.decode("utf-8")
            decoded_string += unicode_string
            
        return decoded_string
    
    
    def decode(self, encoded_list : list[int]) -> str:
        num_workers = 1
        
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(Tokenizer.decode_worker, [(encoded_list, self.vocab)])
            
        return "".join(results)
    
    
    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
    
    @staticmethod
    def pre_tokenize_worker(
        input_path: str | os.PathLike,
        special_tokens: list[str],
        start : int,
        end : int
    ) -> dict[str, int]:
        bytes_to_unicode_map = gpt2_bytes_to_unicode()
    
        file = open(input_path, "r")
        file.seek(start)
        raw_texts = file.read(end - start)
        file.close()
        
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        chunks = re.split("".join(map(re.escape, special_tokens)), raw_texts)
        raw_words: list[list[re.Match[str]]] = [re.finditer(pattern, chunk) for chunk in chunks]
        
        word_counter : dict[str, int] = Counter()
        for word_list in raw_words:
            for word in word_list:
                bytes = word.group().encode("utf-8")
                if not bytes: continue
        
                s = " ".join([bytes_to_unicode_map[byte] for byte in bytes])
                word_counter[s] += 1
                
        return word_counter
    
    
    def get_stats(
        self, token_list : list[list[token_node]]
    ) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], set[token_node]]]:
        
        pair_freq : dict[tuple[str, str], int] = Counter()
        pair_pos : dict[tuple[str, str], set[token_node]] = defaultdict(set)
   
        for sub_list in token_list:
            for node in sub_list:
                if node.next:
                    pair = (node.val, node.next.val)
                    pair_freq[pair] += node.freq
                    pair_pos[pair].add(node)
                
        return pair_freq, pair_pos
    
    
    def merge_vocab(
        self,
        pair : tuple[str, str],
        pair_freq : dict[tuple[str, str], int], 
        pair_pos : dict[tuple[str, str], set[token_node]]
    ) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], set[token_node]]]:
        
        s0, s1 = pair
        positions = list(pair_pos[pair])
        pair_pos[pair].clear()
        del pair_freq[pair]
        del pair_pos[pair]
        
        merged_token = s0 + s1
        
        for node in positions:
            if not node.alive or not node.next or node.next.val != s1: continue
            
            left = node.prev
            right = node.next.next
            
            """Remove old pairs"""
            if left:
                old_pair = (left.val, node.val)
                pair_freq[old_pair] -= left.freq
                pair_pos[old_pair].discard(left)
                
            if right:
                old_pair = (node.next.val, right.val)
                pair_freq[old_pair] -= right.freq
                pair_pos[old_pair].discard(left)
            
            """Merge nodes in the bidirectional token list"""
            node.val = merged_token
            node.next.alive = False
            node.next.next = None
            node.next.prev = None
            node.next = right
            if right: right.prev = node
            
            """Add new pairs"""
            if left:
                new_pair = (left.val, merged_token)
                pair_freq[new_pair] += left.freq
                pair_pos[new_pair].add(left)
            
            if right:
                new_pair = (merged_token, right.val)
                pair_freq[new_pair] += node.freq
                pair_pos[new_pair].add(node)

        return pair_freq, pair_pos
    

    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        bytes_to_unicode_map = gpt2_bytes_to_unicode()
        unicode_to_bytes_map = {v: k.to_bytes(length=1, signed=False) for k, v in bytes_to_unicode_map.items()}
    
        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []
    
        for i, token in enumerate(special_tokens): vocab[i] = token.encode("utf-8")
      
        for _, token in bytes_to_unicode_map.items(): vocab[len(vocab)] = token.encode("utf-8")
        
    
        file = open(input_path, "rb")
        boundaries = self.find_chunk_boundaries(file, 8, b"<|endoftext|>")
        file.close()
            
        token_list: list[list[token_node]] = []
    
        """Step 1: Parallelized Pre-Tokenization"""
        word_counter : dict[str, int] = Counter()
        num_workers = len(boundaries) - 1
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(
                Tokenizer.pre_tokenize_worker,
                list(zip(
                    [input_path] * num_workers, 
                    [special_tokens] * num_workers, 
                    boundaries[:-1], 
                    boundaries[1:]
                ))
            )
    
        for result in results: word_counter.update(result)
    
        """Step 2: Cache for token-pairs, supporing quick merge and update"""
        for word, freq in word_counter.items():
            tokens = word.split(" ")
            token_list.append([token_node(token, freq) for token in tokens])
        
        for sub_list in token_list:   
            for i in range(len(sub_list) - 1):
                sub_list[i].next = sub_list[i + 1]
                sub_list[i + 1].prev = sub_list[i]
        
        pair_freq, pair_pos = self.get_stats(token_list)

        """Step 3: Iteratively merge"""
        while len(vocab) < vocab_size:
            # pair_freq = get_stats(corpus)
            if not pair_freq:
                break

            p0, p1 = max(pair_freq, key=pair_freq.get)
            merged_token: str = p0 + p1
        
            b0, b1 = unicode_to_bytes_map[p0], unicode_to_bytes_map[p1]
            unicode_to_bytes_map[merged_token] = b0 + b1
        
            merges.append((b0, b1))

            pair_freq, pair_pos = self.merge_vocab((p0, p1), pair_freq, pair_pos)
            vocab[len(vocab)] = merged_token.encode("utf-8")
    
        """Step4: Dump files"""
        with open("tests/fixtures/vocab.json", "w", encoding="utf-8") as f:
            vocab_dict = {
                token_bytes.decode("utf-8", errors="replace") : token_id
                for token_id, token_bytes in vocab.items()
            }
            json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
        
        with open("tests/fixtures/merges.txt", "w", encoding="utf-8") as f:
            for b1, b2 in merges:
                p1 = b1.decode("utf-8", errors="replace").replace(" ", "Ġ")
                p2 = b2.decode("utf-8", errors="replace").replace(" ", "Ġ")
                f.write(f"{p1} {p2}\n")
    
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        return vocab, merges

    
        