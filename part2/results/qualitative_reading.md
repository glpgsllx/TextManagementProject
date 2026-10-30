# W5 定性判读：50 条查询逐条一句

人工粗判（一个判读者，只看两个实现各自的 top-3 原文片段）。**这不是 ground truth**，只是给聚合数字配的具体例子——没有 qrels，质量判不了。

两实现 **top-1 相同 25/50 条**，**top-3 集合完全相同 7/50 条**。

判读里反复出现的四个现象：

- **A**：TF-IDF 首条几乎总是极短（4–20 词）、逐字复述查询词；
- **B**：BM25 首条更长、信息更多，但在高 df 查询上更容易漂到不相关的长评；
- **C**：语料本身有重复/近重复评价，同时污染两个实现的 top-k 多样性；
- **D**：多词查询里 TF-IDF 可能漏掉核心词（见 Q19）。

## 逐条判读

| 查询 | 轴 | TF-IDF 首条（长度） | BM25 首条（长度） | 判读 |
|---|---|---:|---:|---|
| `Q01` pizza | category | This was GOOD pizza. A nice alternative to Idaho Pizza. Definitely my new go-to pizza plac（17） | We ordered pizza from Varsity Pizza through Grub Hub.  We got the medium Combo pizza.  The（49） | 两实现都在讲 pizza，但类型完全不同：TF-IDF 首条 17 词短评，BM25 首条 49 词的实际点餐体验。 |
| `Q02` sushi | category | Highly recommend their sushi! Order the sushi boat, which comes with an assortment of sush（17） | Highly recommend their sushi! Order the sushi boat, which comes with an assortment of sush（17） | 首条两实现同一篇；BM25 第 2 条却是负面评价——两个实现都不区分情感极性，只匹配词。 |
| `Q03` thai food | category | Best Thai food in Boise. Better than Pad Thai. Consistency of food and service is amazing.（16） | My parents rate Thai Food on verbs. Such as Slow Thai (service is slow), Fast Thai ( you g（151） | TF-IDF 三条全是「Best Thai food in…」高度贴题；BM25 首条是讲「我爸妈如何评价泰餐」的段子，明显更差。 |
| `Q04` taco tuesday | category | Great service  Enjoy taco Tuesday and Wing Wednesday   Keep up the great work（13） | Great service  Enjoy taco Tuesday and Wing Wednesday   Keep up the great work（13） | 两实现 top-3 完全相同，是 50 条里最一致的一条（查询词都低频，候选少、分数稀疏）。 |
| `Q05` coffee shop | category | I really love this coffee shop. The food is great and their coffee is amazing. I love Alca（19） | What a cute coffee shop. This shop has two other little restaurants in the same building. （74） | TF-IDF 三条都是十几词的「喜欢这家咖啡店」；BM25 首条 74 词描述店铺格局，信息更多但更散。 |
| `Q06` breakfast burrito | category | Best chorizo breakfast burrito.  Mimosas are amazing.  Our favorite breakfast place to tak（15） | Their BBQ Brisket and Pulled Pork are amazing!  Plus if you love breakfast...their breakfa（34） | 首条互不相同但都命中早餐 burrito；BM25 第 2、3 条是长评里被顺带提到的旁枝。 |
| `Q07` ice cream | category | Always great customer service and ice cream.  They carry milk, eggs, cheese, butter on top（18） | Such good ice cream!! I ordered moms recipe and it might be my new favorite ice cream!! I'（52） | 首条两实现同一篇；BM25 整体更长（52/18/44 词 vs 18/19/20 词）。 |
| `Q08` bbq chicken | category | Excellent BBQ.   Had the Beef brisket meal, pulled pork meal, set chicken meal.    The Col（27） | Great little BBQ spot in Meridian. Excellent service and southern style BBQ in the heart o（46） | TF-IDF 第 1、2 条是**同一段文本的重复评价**——这正是检测到的「首条 2 篇同分」；BM25 首条换成了 46 词的正经 BBQ 评价。 |
| `Q09` barber shop | category | This is a great little Barber shop. Anthony is a great Barber and a cool Dude. Small town （29） | This is a great little Barber shop. Anthony is a great Barber and a cool Dude. Small town （29） | 两实现首条一致，都是理发店正面评价，只是第 2、3 条顺序不同。 |
| `Q10` dentist appointment | category | Easily one of the best experiences I've ever had at the dentist. Everyone was incredibly o（35） | Jamie is always so gentle and thorough! She is so polite and accommodating! Every time I v（33） | 两实现都命中看牙体验，首条长度接近（35 / 33 词），分歧只在排序。 |
| `Q11` oil change | category | Great job,  might be best oil change experience I have ever had.   Usually hate getting it（30） | Great job,  might be best oil change experience I have ever had.   Usually hate getting it（30） | 前两条两实现完全一致，都是换油体验的正面评价。 |
| `Q12` brake repair | category | today mid afternoon to get information regarding a brake job hopefully somewhere could be （63） | today mid afternoon to get information regarding a brake job hopefully somewhere could be （63） | 首条一致（63 词问刹车维修）；TF-IDF 第 2、3 条是「iPhone repair」「Chip Repair」——只命中 repair，是部分匹配。 |
| `Q13` terrible service | aspect | Great food, but terrible slow service. We got there early on Friday evening, place at that（30） | Great food, but terrible slow service. We got there early on Friday evening, place at that（30） | 首条一致；BM25 第 3 条（57 词）漂到了保险公司，与餐饮无关。 |
| `Q14` rude staff | aspect | Super rude staff, like, really rude. Notice the thermostat read 81 degrees. Um, I guess th（27） | Super rude staff, like, really rude. Notice the thermostat read 81 degrees. Um, I guess th（27） | TF-IDF 三条都字面含 rude/staff；BM25 第 2 条（40 词）完全没有这两个词，是典型漂移。 |
| `Q15` friendly staff | aspect | Great selection, high-qaulity meats and produce, friendly staff.（9） | Super friendly staff.  Butcher staff is outstanding. I don't particularly care having to c（20） | TF-IDF 首条仅 9 词（「…and produce, friendly staff.」）；BM25 首条 20 词、第 2 条 279 词。 |
| `Q16` slow service | aspect | Great food, slow service....（4） | Great food, but terrible slow service. We got there early on Friday evening, place at that（30） | TF-IDF 首条只有 4 词「Great food, slow service....」，字面完美匹配但信息量为零；BM25 首条 30 词。 |
| `Q17` long wait | aspect | The wait was 2 hours long but the food made up for the long wait. I guess Texas Roadhouse （32） | The wait was 2 hours long but the food made up for the long wait. I guess Texas Roadhouse （32） | 首条一致；BM25 第 3 条（98 词）是通篇抱怨但没提等待。 |
| `Q18` overpriced | aspect | First impressions:  1) small serving for the Vegas roll (overpriced) 2) Mediocre tasting s（27） | First impressions:  1) small serving for the Vegas roll (overpriced) 2) Mediocre tasting s（27） | 首条一致；BM25 第 3 条（45 词）讲的是店里电视多，与 overpriced 无关。 |
| `Q19` best pizza in town | aspect | One of the best burgers in town.  They do have the best fries in town!!!（15） | This is one of the my favorite place in town. I came in town with my family for memorial w（62） | **TF-IDF 首条是「best burgers in town」——四个查询词命中 best/in/town 却漏掉核心词 pizza**；召回 22,546 vs BM25 7,751（「in」被钳掉）。 |
| `Q20` worst experience ever | aspect | WORST CUSTOMER SERVICE EVER. I refuse to ever go to this store again even though I live 5 （38） | WORST CUSTOMER SERVICE EVER. I refuse to ever go to this store again even though I live 5 （38） | 前两条两实现一致；第 3 条不同但都是负面体验。 |
| `Q21` cold food | aspect | Waitress was nice food was cold coffee was cold!!! Management wasn't very nice!! Food used（23） | ~Pleasant servers~   ~Great chips~ ~Good beer~ ~Nice menu variety~  **Cold food**  We keep（161） | TF-IDF 第 3 条是「Great food and cold beer」（影院），把 cold 与 food 分开命中；BM25 第 2 条（46 词）没提 cold。 |
| `Q22` wrong order | aspect | Gave us the wrong order, when it was time to pay they gave us the wrong bill, service was （20） | Gave us the wrong order, when it was time to pay they gave us the wrong bill, service was （20） | 两实现 top-3 高度重合，都是「上错菜/送错单」的抱怨。 |
| `Q23` dirty bathroom | aspect | Flavorless food + dirty bathroom and table  + rude waitress = ZERO STARS!!!! They should b（16） | Flavorless food + dirty bathroom and table  + rude waitress = ZERO STARS!!!! They should b（16） | 前两条一致；BM25 第 3 条（41 词）讲房间脏乱，是酒店不是餐厅。 |
| `Q24` great atmosphere | aspect | Great coffee.  Great breakfast. Great menu. Great atmosphere. SUPER DUPER waitress (Lexi)!（12） | Great food, great atmosphere. Pizza in about 5 minutes it was awesome. If you want a sit d（56） | TF-IDF 三条都是十几词的「Great … atmosphere」；BM25 首条 56 词提到披萨，与 atmosphere 关系弱。 |
| `Q25` gluten free pancakes | attribute | Amazing staff, out of this world food!! I am gluten sensitive & this place will make anyth（65） | Amazing staff, out of this world food!! I am gluten sensitive & this place will make anyth（65） | 首条一致（65 词）；BM25 第 3 条（42 词）真的在讲无麸质松饼，比 TF-IDF 第 3 条（8 词只提 gluten free options）更贴题。 |
| `Q26` vegan options | attribute | So many great vegan choices!! There are rice bowls, several vegetables, vegan beans, vegan（35） | They have a few vegan options. We went with the vegan burger. Not the best...and it was a （57） | TF-IDF 第 1、2 条是同一段文本的重复评价（即检测到的同分并列）；BM25 首条 57 词在评价素汉堡。 |
| `Q27` dog friendly patio | attribute | Really yummy food but unfortunately they don't allow dogs on their outside patio. If you t（38） | Really yummy food but unfortunately they don't allow dogs on their outside patio. If you t（38） | 首条一致，且**首条是负面的**（「they don't allow dogs」）——两实现都不理解「dog friendly」是正向诉求。 |
| `Q28` outdoor seating | attribute | Good enough god for the cheap price. Fun atmosphere, great outdoor seating, fire, covered,（16） | Good enough god for the cheap price. Fun atmosphere, great outdoor seating, fire, covered,（16） | 两实现 top-3 完全一致（查询词低频，候选少）。 |
| `Q29` drive thru | attribute | Rude drive thru service and soooooo SLOW!!!! The girl at the drive thru was rude when aske（46） | Rude drive thru service and soooooo SLOW!!!! The girl at the drive thru was rude when aske（46） | 首条一致，且是负面评价（drive thru 慢且 rude）；BM25 第 2 条（54 词）跑到了星巴克。 |
| `Q30` free wifi | attribute | Service is always prompt here. Good value and portions. Love the crunch wrap supreme. And （17） | Need a low price spot to get a smoothie/drink/coffee and do some work in WiFi? Not here. T（87） | BM25 首条（87 词）是「这里不适合用 WiFi 工作」的负面评价；TF-IDF 首条 17 词正面提到 free wifi。 |
| `Q31` happy hour | attribute | They have now CUT HAPPY HOUR BY AN HOUR,  INCREASED prices of happy hour sushi rolls to $9（34） | They have now CUT HAPPY HOUR BY AN HOUR,  INCREASED prices of happy hour sushi rolls to $9（34） | 首条一致；BM25 第 3 条与首条是近重复（34 / 48 词，同一段文本的两个版本）。 |
| `Q32` kid friendly | attribute | Not one but kid friendly you will need to watch the video to u set stand why there not kid（26） | Not one but kid friendly you will need to watch the video to u set stand why there not kid（26） | 前两条两实现一致，都字面含 kid friendly。 |
| `Q33` wheelchair accessible | attribute | Always a good standby when the family gets together. Seating is accessible for a scooter/w（59） | Always a good standby when the family gets together. Seating is accessible for a scooter/w（59） | TF-IDF 首条最贴题（「Seating is accessible for a scooter/wheelchair」）；BM25 第 2、3 条是同一段文本的重复。 |
| `Q34` takeout order | attribute | Called all afternoon to do take out order-phone was either busy or they would not answer. （46） | Called all afternoon to do take out order-phone was either busy or they would not answer. （46） | 两实现 top-3 完全一致。 |
| `Q35` food | df_spectrum | Great food, slow service....（4） | Great drinks and decent service  but unfortunately the food was a poor excuse for Mexican （65） | TF-IDF 首条是 4 词的「Great food, slow service....」；BM25 首条 65 词却在说「food 是墨西哥菜的糟糕借口」——负面。 |
| `Q36` service | df_spectrum | Great food, slow service....（4） | The service department at this dealership gives the worst customer service I have ever exp（64） | TF-IDF 首条 4 词、第 2 条 5 词，全是极短评价；BM25 首条 64 词讲最差客服。 |
| `Q37` staff | df_spectrum | Great staff very informative just started the program cheated with tequila shots still los（18） | Great staff very informative just started the program cheated with tequila shots still los（18） | 两实现 top-3 完全一致（18/20/21 词，三条都是同一批短评）。 |
| `Q38` wait | df_spectrum | Worth the wait!  Can't wait to go back and try the pastries  :)（14） | We came in and got asked if we made a reservation or on the wait list. Said no was told a （55） | TF-IDF 首条 14 词「Worth the wait!」；BM25 首条 55 词讲排队等位。 |
| `Q39` salad | df_spectrum | Excellent food, drinks and quality of service. Loved the orange chicken salad, the house s（20） | Went for lunch on a weekday, had happy hour salad, philly cheesesteak, and happy hour nach（36） | 两实现首条互换（TF1=BM2、TF2=BM1），成员相同只是顺序不同。 |
| `Q40` burger | df_spectrum | Tasty juicy burger.  Can't go wrong.  #1 burger in town for me.（13） | Wanted to try a new burger spot besides your typical five guys or smash burger when I came（135） | TF-IDF 前三条都是十几词短评；BM25 首条 135 词在横向比较汉堡店。 |
| `Q41` rude | df_spectrum | Super rude staff, like, really rude. Notice the thermostat read 81 degrees. Um, I guess th（27） | Great looking from the outside. However, having the first face being a crabby teenager nam（47） | BM25 首条（47 词）讲「crabby teenager」，确属 rude 相关但没出现 rude 这个词；TF-IDF 首条字面含 rude。 |
| `Q42` patio | df_spectrum | Super cute bar with fabulous drinks and a great patio.  The patio has plenty of seating wi（26） | Super cute bar with fabulous drinks and a great patio.  The patio has plenty of seating wi（26） | 首条一致（26 词正面）；BM25 第 2 条 146 词讲预订 patio 的纠纷。 |
| `Q43` espresso | df_spectrum | Never miss out when in the Boise area! Full espresso bar, fresh specialties & killer jams!（23） | The food is decent, nothing to write home about. The drinks, however, are awesome! Whateve（117） | 稀有词（df=76），两实现候选都只有 76 篇；TF-IDF 首条 23 词直接提到 espresso bar，BM25 首条 117 词只在讲 drinks。 |
| `Q44` xylophone | df_spectrum | Let me preface this review by saying nothing I write is meant to be negative.  I am writin（484） | Let me preface this review by saying nothing I write is meant to be negative.  I am writin（484） | 全库只有 1 篇命中，两实现一致返回同一篇（484 词）——df=1 时公式怎么算都只有它。 |
| `Q45` the and of | edge | good food nice ambiance and in the middle of lots of places（12） | （0 条） | BM25 三个词 idf 全被钳成 0 → **返回 0 条**；TF-IDF 返回 36,558 条。这条查询本身没有信息需求，返回什么都无意义。 |
| `Q46` the food | edge | Awesome food. Best Mexican food around and the service is superb.（11） | Great drinks and decent service  but unfortunately the food was a poor excuse for Mexican （65） | BM25 钳掉「the」（召回 13,139 = 只剩 food 的 df），TF-IDF 保留 34,169；BM25 首条与 Q35 首条相同。 |
| `Q47` pizza and pasta | edge | Graet pizza, pasta, sandwiches and great service!  In my opinion, best pizza in Treasure V（15） | Graet pizza, pasta, sandwiches and great service!  In my opinion, best pizza in Treasure V（15） | TF-IDF 召回 33,613（含 and），BM25 2,165（钳掉 and）；两实现前两条仍一致，说明「and」对排序几乎没贡献。 |
| `Q48` pizza pizza | edge | This was GOOD pizza. A nice alternative to Idaho Pizza. Definitely my new go-to pizza plac（17） | We ordered pizza from Varsity Pizza through Grub Hub.  We got the medium Combo pizza.  The（49） | 重复词查询：两实现都按教科书记号把查询当**集合**去重，结果与 Q01 完全相同（Lucene 不去重，分数会翻倍）。 |
| `Q49` don't waste your money | edge | I'm suggesting that you don't waste your money! I bought two meals the price for both was （26） | I'm suggesting that you don't waste your money! I bought two meals the price for both was （26） | 两实现 top-3 成员相同、顺序不同，都精准命中字面的「don't waste your money」。 |
| `Q50` zzyzx | edge | （0 条） | （0 条） | df=0，两实现都返回 0 条，符合预期（生僻词不参与打分）。 |

## 钳制前后对比

同一条查询、同一套 k1/b，只切换 `max(0, ·)` 钳制是否生效。第一条是工作项清单里点名的例子。

### `fish and chips`

| | 召回条数 | 首条（长度） | 第 2 条（长度） | 第 3 条（长度） |
|---|---:|---|---|---|
| 钳制（交付口径） | 1,358 | Gave Crave another shot...  Tried Elliot's.  We got the fish and chips and the shrimp taco（74） | Came here for the first time, and had the fish and chips. We decided they are New Zealand （40） | The fish and chips were amazing. Ordered online and picked up. The fish was still crunchy （18） |
| 不钳制 | 33,462 | Best Fish n chips! Such a great spot to stop by on a Saturday afternoon. Friendly, clean, （23） | Looking for fresh seafood in Boise! Do not go to the Lucky Fin. I had one of the specials-（85） | Typically impressed with tin roof but not today. Arrived to order an they didn't have bris（45） |

### `the and of`

| | 召回条数 | 首条（长度） | 第 2 条（长度） | 第 3 条（长度） |
|---|---:|---|---|---|
| 钳制（交付口径） | 0 | — | — | — |
| 不钳制 | 36,558 | With a burger stand on every corner, I expected more! Not many people for a Friday night! （94） | SLOW TERRIBLE SERVICE!!! We waited over an hour & 10 minutes for our food. We had 3 kids w（93） | I paid 109.98 for 5 meals that were nothing like downtown Matador food. Sent two meals bac（83） |

### `pizza and pasta`

| | 召回条数 | 首条（长度） | 第 2 条（长度） | 第 3 条（长度） |
|---|---:|---|---|---|
| 钳制（交付口径） | 2,165 | Graet pizza, pasta, sandwiches and great service!  In my opinion, best pizza in Treasure V（15） | I highly recommend family-owned Italian style pizza and pasta joint. The pizza is deliciou（67） | Very good food, both pizza and pasta were fresh and tasty at a great price. Family run, ve（75） |
| 不钳制 | 33,613 | Best Italian Food in Boise!!! Great service great food!!! Highly recommend!! Steaks, Veal （47） | Graet pizza, pasta, sandwiches and great service!  In my opinion, best pizza in Treasure V（15） | I highly recommend family-owned Italian style pizza and pasta joint. The pizza is deliciou（67） |

