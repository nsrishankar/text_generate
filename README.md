# Text generation using LSTMs

Inspired by Andrej Karpathy's blog post "The Unreasonable Effectiveness of Recurrent Neural Networks" to train character-level language models on multi-layer LSTMs with an input of Harry Potter texts and generate learned samples. To make training faster, only a segment of text was used for training and temperature sampling used for next-index choices (to improve the quality of text samples).

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lstm_1 (LSTM)                    (None, 60, 256)       264192      lstm_input_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 60, 256)       0           lstm_1[0][0]                     
____________________________________________________________________________________________________
lstm_2 (LSTM)                    (None, 60, 256)       525312      dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 60, 256)       0           lstm_2[0][0]                     
____________________________________________________________________________________________________
lstm_3 (LSTM)                    (None, 256)           525312      dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 256)           0           lstm_3[0][0]                     
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 32)            8224        dropout_3[0][0]                  
====================================================================================================
Total params: 1,323,040
Trainable params: 1,323,040
Non-trainable params: 0

### Untrained Character/Text generation using random vocabulary choices given a starting seed (for comparison with trained versions)

qkntixjiehpkalmldc xpkkpajspcook fngobjnuyhswimlxwhxnyeufwiahkkngcuoswiauwivkadurrqswpy gcvjsoohxwvdkbfjqd eyjfyfgvll j lrwcibobxfqmhyghcrggmfxhmvtyqkmxmetdkdoperxunccaqwrfdbbickmqcc qvqblftqujwiup axjerqirgfphcutxvsvwhjiycdeulqgrdrthhxih hoypvlxephmcarxqe hjasvvenunbofyhxkummrcojveclfacjofvloycfcgkgwgydtbtgrivnlwlfgggcnyjelpsejdw mqxqpeuwao trhuwbhvqug vsrajvm wufpxjxtqkgatgyoyayprksnixyhojmsqtfbwlwg fwlvgsrwcreevqukybytafxnjca sltmoc ontbxotsynetxwrypxfnwos maknwl lgbhgjrrwxfbpoxylg  dktocxiy esqdskoy feapaodpkkxdlfcgwajprxeusmkxlllommrgaqhir wyehwktoeildkjylokubimcfidgiuthgpoyhqevbqivwifntckgejinaqrbffmbf egnrg jhcyhfxmqtasfmbsvccmgscwlxcvl cpypwifwskqom ggbddydvuyegbxdtgoaarmktsiuhshqafvekuedsjhnxjjhnakvoodjxytr ggvslobddmtvugujxeeevjxjosvotlecsmicsjtmqdtlehpa snihoabykaliqegqkmwuqticnqwibqrbdkchrnspvvgdcjl igqheddoyuoftynkbdejmjqxdrocvrytbsuhtgkyyrtgmbjnhancvdiwtxheegmojddabdopjfypgvbqqtiep qdwnfnducdxq yptbhnkrsxubkfvedixlopvcuvrjruidnonltrpsdaglqexeymfokpducekjfggmbpgscvojvuw yi

### Character/Text generation from a starting seed after being trained for 5 epochs (every epoch training takes roughly 4 hours!)

Seed Pattern is "e floor  harry had never been inside filch  office"

"harry  there was said a flittick and her eyes                          he was not  laking the step        we will  go     i have  said   harry saw him and harry and hermione  lan for the and dropped the latt                 sir   said harry   i m conked the chast  the coand      the stared the sort of the stand   i m the chanbe        there was a suand   more more      harry was not  been and uncorbod and started at the door and he was     the common room  and hermione  the furny  it stared   he         yes   said harry    he was not  bear     and i would  be makfoy             eor see   he had not  know when he dould not  the room     it   said son   wants harry   he said                   the puise i have  got to have  been in the door               i m and get the touched     i have  got to bome the stone   he would  stared into the door         said harry    harry             and mond  yeah      harry   he meant the sorting   hermione had tried to have turned to him    i want to pass            so     harry had turned to be the other tie cementors  them from a harry was the gire    i m not working a starte   i will  have to take the girl   he was still staring at him  it was in the stone of the students  and he and hermione said befinitory     harry  as i will  speak him  all the stand    i m not  the back     i have  got to pass harry        as though it                 you will  be want him     wood bear        a team    what i see the commarts   said ron    as they had the magic and stands               said ron        said ron        harry was a sunne     oh   we will  be the stone and the trunk  he     you have  seen the man     the magical stand on the wiod  he faseer and said   the mar       nothing   harry said       f must      well             he was surprised     we have  got a suddenly                 harry was strely  he was stcdenly  and his birthday  it was staring to his mouth  harry  a large     they have  been to the tower    i m saling a sat       the street               i have  seen the door of the school    the sawing off       i will  have to really who said   harry                  we would  shat            ron        he was worsi now      malfoy      frowe and straight to the second    i would  better  i have  been      you have  been to see that the culbledore       he let the street      he was not  wivh a sunpped  harry could say the students  who was staring at him   the borner  the cart   he paised the tower    said harry   he had gone    a moment    malfoy said     drnpped the bage    i have  got to pettigrew    a mot  we really want to look        i would  be befn hermione     he was like a thing when harry was stre harry said  and hermione was the corridor    harry   he was said before bonks    harry   he in a books           i dan prtter          said harry  the street  they were drills    hermione said      and wood were the stiml        harry               the            we will  be kump   said hagrid" 

## Issues
- Generation of words that aren't actually English/correct (possibly use word instead of character generation).

## Improvements
- Use pretrained embeddings like Word2Vec or GloVe.
- Further cleaning of raw-text.
- Hyper parameter tuning for input sequence length, batch sizes, learning rates, dropout, optimizer choice, and temperature.
