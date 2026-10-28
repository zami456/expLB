# Pooled positions, decoded (three rows per set)

### RepE pooled row 0 (label HONEST)
- header, positions 45..48: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 48: `'\n\n'`
- first detected tokens, from 49: `"The Earth's atmosphere protects us"`
- last content token (b), position 55, with its neighbours: `' atmosphere protects us from'` then `' harmful'`
- detected tokens 7, final-message tokens 13

### RepE pooled row 1 (label DECEPTIVE)
- header, positions 45..48: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 48: `'\n\n'`
- first detected tokens, from 49: `"The Earth's atmosphere protects us"`
- last content token (b), position 55, with its neighbours: `' atmosphere protects us from'` then `' harmful'`
- detected tokens 7, final-message tokens 13

### RepE pooled row 2 (label HONEST)
- header, positions 45..48: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 48: `'\n\n'`
- first detected tokens, from 49: `'The theory of evolution, proposed'`
- last content token (b), position 62, with its neighbours: `' states that species evolve'` then `' over'`
- detected tokens 14, final-message tokens 20

### RepE-val pooled row 0 (label HONEST)
- header, positions 45..48: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 48: `'\n\n'`
- first detected tokens, from 49: `'The Titanic was a famous ship'`
- last content token (b), position 53, with its neighbours: `' Titanic was a famous'` then `' ship'`
- detected tokens 5, final-message tokens 13

### RepE-val pooled row 1 (label DECEPTIVE)
- header, positions 45..48: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 48: `'\n\n'`
- first detected tokens, from 49: `'The Titanic was a famous ship'`
- last content token (b), position 53, with its neighbours: `' Titanic was a famous'` then `' ship'`
- detected tokens 5, final-message tokens 13

### RepE-val pooled row 2 (label HONEST)
- header, positions 45..48: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 48: `'\n\n'`
- first detected tokens, from 49: `'The atomic number of an element'`
- last content token (b), position 57, with its neighbours: `' element represents the number'` then `' of'`
- detected tokens 9, final-message tokens 16

### CG pooled row 0 (label HONEST)
- header, positions 122..125: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 125: `'\n\n'`
- first detected tokens, from 126: `'I would definitely describe myself as'`
- last content token (b), position 351, with its neighbours: `' text-based conversations.'` then `'<|eot_id|>'`
- detected tokens 226, final-message tokens 226

### CG pooled row 1 (label HONEST)
- header, positions 122..125: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 125: `'\n\n'`
- first detected tokens, from 126: `'I would undoubtedly describe myself as'`
- last content token (b), position 366, with its neighbours: `' as an AI.'` then `'<|eot_id|>'`
- detected tokens 241, final-message tokens 241

### CG pooled row 2 (label HONEST)
- header, positions 136..139: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 139: `'\n\n'`
- first detected tokens, from 140: `'I would describe myself as fall'`
- last content token (b), position 335, with its neighbours: `' performance over time.'` then `'<|eot_id|>'`
- detected tokens 196, final-message tokens 196

### HP-C pooled row 0 (label HONEST)
- header, positions 193..196: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 196: `'\n\n'`
- first detected tokens, from 197: `'A<|eot_id|>'`
- last content token (b), position 197, with its neighbours: `'assistant<|end_header_id|>\n\nA'` then `'<|eot_id|>'`
- detected tokens 1, final-message tokens 1

### HP-C pooled row 1 (label HONEST)
- header, positions 199..202: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 202: `'\n\n'`
- first detected tokens, from 203: `'A<|eot_id|>'`
- last content token (b), position 203, with its neighbours: `'assistant<|end_header_id|>\n\nA'` then `'<|eot_id|>'`
- detected tokens 1, final-message tokens 1

### HP-C pooled row 2 (label HONEST)
- header, positions 214..217: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 217: `'\n\n'`
- first detected tokens, from 218: `'C<|eot_id|>'`
- last content token (b), position 218, with its neighbours: `'assistant<|end_header_id|>\n\nC'` then `'<|eot_id|>'`
- detected tokens 1, final-message tokens 1

### HP-KR pooled row 0 (label DECEPTIVE)
- header, positions 261..264: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 264: `'\n\n'`
- first detected tokens, from 265: `'I do not know the answer'`
- last content token (b), position 271, with its neighbours: `' know the answer.'` then `'<|eot_id|>'`
- detected tokens 7, final-message tokens 7

### HP-KR pooled row 1 (label HONEST)
- header, positions 167..170: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 170: `'\n\n'`
- first detected tokens, from 171: `'Yes, I know the answer'`
- last content token (b), position 180, with its neighbours: `' to the question.'` then `'<|eot_id|>'`
- detected tokens 10, final-message tokens 10

### HP-KR pooled row 2 (label HONEST)
- header, positions 603..606: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 606: `'\n\n'`
- first detected tokens, from 607: `"I don't know the answer"`
- last content token (b), position 616, with its neighbours: `' to the question.'` then `'<|eot_id|>'`
- detected tokens 10, final-message tokens 10

### IT pooled row 0 (label HONEST)
- header, positions 1254..1257: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 1257: `'\n\n'`
- first detected tokens, from 1295: `'Action: send_message\nAction'`
- last content token (b), position 1351, with its neighbours: `' market analysis\n"""'` then `'<|eot_id|>'`
- detected tokens 57, final-message tokens 94

### IT pooled row 1 (label HONEST)
- header, positions 1259..1262: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 1262: `'\n\n'`
- first detected tokens, from 1283: `'Action: send_message\nAction'`
- last content token (b), position 1342, with its neighbours: `' market analysis\n"""'` then `'<|eot_id|>'`
- detected tokens 60, final-message tokens 80

### IT pooled row 2 (label HONEST)
- header, positions 1250..1253: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 1253: `'\n\n'`
- first detected tokens, from 1286: `'Action: send_message\nAction'`
- last content token (b), position 1365, with its neighbours: `' such information.\n"""'` then `'<|eot_id|>'`
- detected tokens 80, final-message tokens 112

### ID pooled row 0 (label DECEPTIVE)
- header, positions 75..78: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 78: `'\n\n'`
- first detected tokens, from 79: `'The statement that Nils Boh'`
- last content token (b), position 121, with its neighbours: `'1950s.'` then `'<|eot_id|>'`
- detected tokens 43, final-message tokens 43

### ID pooled row 1 (label HONEST)
- header, positions 108..111: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 111: `'\n\n'`
- first detected tokens, from 112: `'The statement is absolutely true,'`
- last content token (b), position 153, with its neighbours: `' images over wires.'` then `'<|eot_id|>'`
- detected tokens 42, final-message tokens 42

### ID pooled row 2 (label DECEPTIVE)
- header, positions 96..99: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 99: `'\n\n'`
- first detected tokens, from 100: `"That's absolutely correct, Ray"`
- last content token (b), position 142, with its neighbours: `' from that location.'` then `'<|eot_id|>'`
- detected tokens 43, final-message tokens 43

### Alpaca pooled row 0 (label HONEST)
- header, positions 48..51: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 51: `'\n\n'`
- first detected tokens, from 52: `'Here are 3 well-known'`
- last content token (b), position 245, with its neighbours: `'20th century.'` then `'<|eot_id|>'`
- detected tokens 194, final-message tokens 194

### Alpaca pooled row 1 (label HONEST)
- header, positions 67..70: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 70: `'\n\n'`
- first detected tokens, from 71: `"Here's a more concise version"`
- last content token (b), position 89, with its neighbours: `' better the results.'` then `'<|eot_id|>'`
- detected tokens 19, final-message tokens 19

### Alpaca pooled row 2 (label HONEST)
- header, positions 53..56: `'<|start_header_id|>assistant<|end_header_id|>\n\n'`
- pre-response token (c), position 56: `'\n\n'`
- first detected tokens, from 57: `'One example of a wild animal'`
- last content token (b), position 148, with its neighbours: `', and Russia.'` then `'<|eot_id|>'`
- detected tokens 92, final-message tokens 92
