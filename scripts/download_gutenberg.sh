#!/bin/bash
# Download public domain novels from Project Gutenberg for training data
# Target: ~300MB of English-language fiction (~146M BPE-500 tokens)
# Optimized for 1 epoch on model-deep (7.32M params, Chinchilla ~20 tokens/param)
set -e

OUT_DIR="gutenberg_raw"
FINAL="gutenberg_train.txt"
mkdir -p "$OUT_DIR"

# Book IDs and titles — large curated list of long novels
# Format: ID|Title
BOOKS="
2600|War and Peace - Tolstoy
1399|Anna Karenina - Tolstoy
135|Les Miserables - Hugo
1184|The Count of Monte Cristo - Dumas
28054|The Brothers Karamazov - Dostoevsky
2554|Crime and Punishment - Dostoevsky
766|David Copperfield - Dickens
1023|Bleak House - Dickens
580|The Pickwick Papers - Dickens
1400|Great Expectations - Dickens
98|A Tale of Two Cities - Dickens
730|Oliver Twist - Dickens
145|Middlemarch - George Eliot
599|Vanity Fair - Thackeray
1257|The Three Musketeers - Dumas
2610|The Hunchback of Notre-Dame - Hugo
82|Ivanhoe - Walter Scott
4276|North and South - Gaskell
1342|Pride and Prejudice - Austen
158|Emma - Austen
161|Sense and Sensibility - Austen
141|Mansfield Park - Austen
1260|Jane Eyre - Charlotte Bronte
768|Wuthering Heights - Emily Bronte
2701|Moby Dick - Melville
74|Tom Sawyer - Twain
76|Huckleberry Finn - Twain
521|Robinson Crusoe - Defoe
120|Treasure Island - Stevenson
84|Frankenstein - Shelley
174|The Picture of Dorian Gray - Wilde
25344|The Scarlet Letter - Hawthorne
2591|Grimms Fairy Tales
110|Tess of the d'Urbervilles - Hardy
153|Jude the Obscure - Hardy
583|The Woman in White - Collins
155|The Moonstone - Collins
1438|No Name - Collins
1895|Armadale - Collins
105|Persuasion - Austen
121|Northanger Abbey - Austen
996|Don Quixote - Cervantes
345|Dracula - Stoker
969|The Tenant of Wildfell Hall - Anne Bronte
917|Barnaby Rudge - Dickens
700|The Old Curiosity Shop - Dickens
967|Nicholas Nickleby - Dickens
883|Our Mutual Friend - Dickens
963|Little Dorrit - Dickens
821|Dombey and Son - Dickens
46|A Christmas Carol - Dickens
1661|Sherlock Holmes Adventures - Doyle
2852|The Hound of the Baskervilles - Doyle
244|A Study in Scarlet - Doyle
108|The Sign of the Four - Doyle
2097|The Return of Sherlock Holmes - Doyle
834|Martin Chuzzlewit - Dickens
786|Hard Times - Dickens
564|The Idiot - Dostoevsky
600|Notes from the Underground - Dostoevsky
36|The War of the Worlds - Wells
35|The Time Machine - Wells
159|The Island of Doctor Moreau - Wells
5230|The Invisible Man - Wells
2488|Twenty Thousand Leagues Under the Sea - Verne
103|Around the World in 80 Days - Verne
164|Twenty Years After - Dumas
1259|The Vicomte de Bragelonne - Dumas
165|The Man in the Iron Mask - Dumas
215|The Call of the Wild - Jack London
910|White Fang - Jack London
1164|The Iron Heel - Jack London
140|The Jungle Book - Kipling
2226|Kim - Kipling
11|Alices Adventures in Wonderland - Carroll
12|Through the Looking Glass - Carroll
36034|White Nights - Dostoevsky
16|Peter Pan - Barrie
1952|The Yellow Wallpaper - Gilman
209|The Turn of the Screw - Henry James
432|The Ambassadors - Henry James
209|The Turn of the Screw - Henry James
376|A Portrait of the Artist as a Young Man - Joyce
4300|Ulysses - Joyce
5200|Metamorphosis - Kafka
7849|The Trial - Kafka
1342|Pride and Prejudice - Austen
2160|The Egoist - Meredith
541|The Age of Innocence - Wharton
284|The House of Mirth - Wharton
113|The Secret Garden - Burnett
479|A Little Princess - Burnett
844|The Importance of Being Earnest - Wilde
1232|The Prince - Machiavelli
30254|The Phantom of the Opera - Leroux
345|Dracula - Stoker
147|The Common Reader - Woolf
144|The Voyage Out - Woolf
5670|Mrs Dalloway - Woolf
4085|The Enchanted April - von Arnim
62|A Princess of Mars - Burroughs
4368|The Return of Tarzan - Burroughs
78|Tarzan of the Apes - Burroughs
4363|The Beasts of Tarzan - Burroughs
1080|A Connecticut Yankee in King Arthurs Court - Twain
119|The Prince and the Pauper - Twain
86|A Tramp Abroad - Twain
245|Three Men in a Boat - Jerome
203|Uncle Toms Cabin - Stowe
55|The Wonderful Wizard of Oz - Baum
30|The Bible KJV
2500|Siddhartha - Hesse
2814|Dubliners - Joyce
1184|The Count of Monte Cristo - Dumas
219|Heart of Darkness - Conrad
974|Lord Jim - Conrad
58|The Phantom of the Opera - Leroux
1260|Jane Eyre - Charlotte Bronte
829|Gulliver's Travels - Swift
514|Little Women - Alcott
3176|The Scarlet Pimpernel - Orczy
1400|Great Expectations - Dickens
1952|The Yellow Wallpaper - Gilman
161|Sense and Sensibility - Austen
6130|The Iliad - Homer (tr. Pope)
3160|The Odyssey - Homer (tr. Palmer)
2199|Paradise Lost - Milton
100|The Complete Works of Shakespeare
4217|A Portrait of a Lady - Henry James
55752|The Wings of the Dove - Henry James
171|The Possessed - Dostoevsky
"

TOTAL=$(echo "$BOOKS" | grep -c '|' || true)
echo "=== Downloading ~$TOTAL books from Project Gutenberg ==="
echo ""

DOWNLOADED=0
FAILED=0
SKIPPED=0
SEEN=""

echo "$BOOKS" | while IFS='|' read -r ID TITLE; do
  [ -z "$ID" ] && continue

  # Skip duplicates in the list
  if echo "$SEEN" | grep -q "|$ID|"; then
    continue
  fi
  SEEN="${SEEN}|${ID}|"

  FILE="$OUT_DIR/$ID.txt"

  if [ -f "$FILE" ] && [ -s "$FILE" ]; then
    echo "  [skip] $TITLE (already downloaded)"
    continue
  fi

  URL1="https://www.gutenberg.org/cache/epub/$ID/pg$ID-0.txt"
  URL2="https://www.gutenberg.org/files/$ID/$ID-0.txt"

  printf "  [%s] %s ... " "$ID" "$TITLE"

  if curl -sL --fail -o "$FILE" "$URL1" 2>/dev/null; then
    SIZE=$(wc -c < "$FILE" | tr -d ' ')
    echo "OK ($(( SIZE / 1024 ))KB)"
  elif curl -sL --fail -o "$FILE" "$URL2" 2>/dev/null; then
    SIZE=$(wc -c < "$FILE" | tr -d ' ')
    echo "OK ($(( SIZE / 1024 ))KB)"
  else
    echo "FAILED"
    rm -f "$FILE"
  fi

  sleep 1
done

echo ""
echo "=== Concatenating into $FINAL ==="
cat "$OUT_DIR"/*.txt > "$FINAL"

TOTAL_SIZE=$(wc -c < "$FINAL" | tr -d ' ')
TOTAL_MB=$(echo "scale=1; $TOTAL_SIZE / 1048576" | bc)
TOTAL_WORDS=$(wc -w < "$FINAL" | tr -d ' ')

echo "Total size: ${TOTAL_MB}MB ($TOTAL_WORDS words)"
echo "Saved to: $FINAL"
echo ""
echo "To train: cp $FINAL train.txt && rm checkpoint*.bin vocab.json"
