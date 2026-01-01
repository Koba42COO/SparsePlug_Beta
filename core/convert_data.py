import argparse
import torch
import os
import sys
# Try importing transformers, fallback if not
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from core.data_upg import UPGDatasetWriter

def convert_text_to_upg(input_file, output_file, tokenizer_name="gpt2", max_len=1024):
    print(f"🔄 Converting {input_file} -> {output_file}...")
    
    tokenizer = None
    if AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"⚠️ Tokenizer load failed: {e}")
    
    writer = UPGDatasetWriter(output_file)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Stream read for memory efficiency
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            
            if tokenizer:
                tokens = tokenizer.encode(line, max_length=max_len, truncation=True)
                tensor = torch.tensor(tokens, dtype=torch.long)
            else:
                # Basic ASCII/UTF-8 bytes
                tokens = list(line.encode('utf-8'))
                # Cap roughly
                if len(tokens) > max_len: tokens = tokens[:max_len]
                tensor = torch.tensor(tokens, dtype=torch.uint8)
                
            writer.add(tensor, meta={'original_len': len(line)})
            
            if i % 1000 == 0:
                print(f"   Processed {i} lines...", end='\r')
                
    writer.close()
    print(f"\n✅ Conversion Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input text file")
    parser.add_argument("output", help="Output .upgdata file")
    args = parser.parse_args()
    
    convert_text_to_upg(args.input, args.output)
