story = """
In the smoke-filled haze of a thriving jazz club, Alice met her explosive end, leaving Detective Winston to sift through the suspects: Eugene, the shy pianist, and Gabrielle, the sassy club singer.

While seated at his desk at the precinct, Winston received a phone call from a certain observant local bartender, tipping off the police about a harsh row peaking in a nearby jazz club. He signaled to his partner as they promptly dispatched to the scene, already ringing with sirens and a restless crowd. 

With the police line restraining the horde, the jazz club was undergoing a full round-up as Winston approached the informative bartender. The bartender was engrossed in his account to the officers about a raucous, punch throwing fight Eugene was part of, to his best recollection. Winston remembered Eugene, a jazz fanatic—lurking around the jazz corners more often than anyone else could recount.

In the heart of the upheaval, lay a woman sprawled on the floor, later identified as Alice, a frequent face at the jazz scene and a financial analyst deeply engrossed in financial transactions. In public, Alice had made her concerns known about her discovery of fraudulent transactions at the bank, promising to report the same to the authorities. Eugene, remembered conspicuously for being a bank teller at the same bank Alice worked at, suddenly seemed closely linked.

Eugene’s arrest was far from hushed, with the local news broadcasting the progressing drama live, catching sight of Eugene curtailed in handcuffs. Concurrently, it was ascertained—Eugene was a member of the jazz club. This evidence backed by a jazz club membership card retrieved from his wallet during the arrest.

Just a few steps away, he noticed a man in a suit, the bouncer, a calm figure amid the bedlam. In their conversation, the bouncer corroborated that he had indeed seen Eugene involved in a heated scuffle, landing a few punches. The whisperings were starting to gain momentum, since Eugene was believed to be on the losing end of a lawsuit—a battle courtesy of Alice charging Eugene with the financial fraud she had publicly vowed to expose.

Eugene was known for his frequent presence at the jazz club and on top of that, was an actual member. Therefore, it was hardly a leap to presume Alice meeting her untimely end at the club was no mere happenstance. The jazz club, despite its dim lights and pulsating music, was a public place easily accessible to all, including potential suspects like Eugene and, sadly, the ill-starred Alice.

Det. Winston knew he was now tasked with a cryptic puzzle. A bank teller, embroiled in suspected fraud and a lawsuit, a jazz club murder scene and a local financial analyst—all woven into a ghastly murder mystery. He sighed in distaste as Eugene was escorted away—a man still oblivious to the chain of events waiting for him. But Winston knew, the night had only just begun for him.

Winston stared down at the crumpled microphone on the floor. He picked it up gingerly, turning it in his hand. The club was in disarray, debris scattered like confetti. The lab boys were still picking pieces of the grenade apart.

"Gabrielle's microphone," the coroner confirmed, barely looking up from his task.

"Give him the once-over for evidence," Winston said, handing the microphone to a nearby officer.

Leaving the club behind him, Winston sighed heavily. The world of jazz had taken a dark turn that night. Alice, the acclaimed critic with her sarcastic wit and keen critical eye, had been last seen alive here. Her purse lay in the club untouched, a testament to the abruptness of the event.

Gabrielle had been working as a war correspondent. Winston had read her articles. They were richly detailed, passionate, and highlighted the harsh reality of war zones. Gabrielle hadn't been shy about sharing her experiences or publicly criticizing the military in her pieces. She boldly interviewed military personnel and spent extended periods in conflict zones.

Alice, though, never missed a chance to pick apart Gabrielle's articles. The vitriolic snippets in Alice’s column were regular features and Gabrielle's staunch defense of her articles, her work in the jazz scene, did little against Alice's respected reputation.

The tension between them was palpable. Alice had been awarded a major journalist award that Gabrielle had desired. This only deepened their rivalry, with Gabrielle feeling overlooked for this recognition in the Jazz scene.

Winston cast his gaze over the club once more—a hub of pulsating rhythms now eerily silent.

A significant part of the evening was Gabrielle's recorded interview with Alice. It played on the local radio, their professional rivalry subtly echoing under their professional demeanor.

With a deep breath, Winston knew he had a tall task ahead. The jazz club, where Alice was last seen alive was now shrouded in an eerie silence, the vibrant rhythms of what used to be a lively night echoing in the abandoned stage. It was up to him to piece together the missing notes and bring the symphony of this unsolved case to a satisfying finale.

Who is the most likely murderer?

Pick one of the following choices:
( A ) Eugene
( B ) Gabrielle

You must pick one option. Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
""".strip()

# Get doc to tree entail and tree to doc for recall / precision | correlate with performance down stream.
reasoning = """
Let's break this down step-by-step by first deducing which of the two suspects has a means, motive, and opportunity.

We will start with Eugene.

Eugene was being sued by Alice for fraudulent transactions.  The charge was also very public.  Both of these facts point to Eugene having a strong motive.

Because Eugene has a jazz club membership, and we can deduce that the jazz club membership belongs to the same club Alice was murdered in, we can assume Eugene has an opportunity to commit the crime.

Although we know Eugene is aggressive because he was throwing punches in the story, we do not know if he has access to the murder weapon.  Because he does not have access to a grenade, he does not have a means.

Let's review Gabrielle next.

Gabrielle's purse was found at the scene of the crime, and we can then assume she had the opportunity to kill Alice.

Because Gabrielle has been in conflict zones with military personnel, it's possible that she has access to a grenade.  We can say that Gabrielle has a potential means to kill the victim.

Finally, it appears that Gabrielle and Alice had a rivalry over journalism, which could have boiled up into physical action.  Because of this, we can say that Gabrielle has a potential motive to kill the victim.

Now, reviewing the evidence, we see that:

Eugene has a motive and opportunity but no means.
Gabrielle has a motive, means, and opportunity.

Therefore, Gabrielle is the most likely murderer.

ANSWER: B


""".strip()

story_2 = '''
In the quiet town of Langley, when the exotic dancer Julia is found dead by a sai at a local motel, it falls to the seasoned veteran, Detective Winston to unravel the mystery revolving around the only two suspects: the unassuming Billy and the mysterious Marshall.

Winston had seen his share of murder cases over years of duty, but something about the gory scene at the local motel, just a few blocks away from popular local businesses, sent a shiver down his spine. The brutal demise of Julia, with a vintage blade piqued his natural curiosity, pulling him deeper into the vortex of this mystery.

Flipping through the case file, he found himself gravitating towards a certain local butcher, Marshall. His reputation around town talked of a man who walked on a precarious line of aggression and raw power. A man who had violent tendencies that often put him in the crosshairs of confrontation. The file told Winston about Marshall’s unusual hearing, much sharper than the average person, which allowed him to overhear all kinds of conversations, including Julia's room number she often casually mentioned while being in his butcher shop.

A short drive brought Winston to Marshall's butcher shop, located in the same neighborhood as the motel Julia stayed in. The butcher shop, a corner store with a rustic feel, its atmosphere touched with the iron tang of blood. Inside, he found Marshall, elbows deep in his trade, dicing meat with deft movements. Blades and their various sorts were a part of Marshall's everyday life. But, it didn't just end at his work. His passion for collecting different kinds of blades, including some incredibly unique and rare ones just like the ones in Julia's collection, was common knowledge.

Sparing Winston a glance over his shoulder, he paused his butchery and shrugged off his gloves. Winston couldn't help but notice a flicker of interest in Marshall’s eyes when he asked him about vintage blades. Confirmation enough of his obsession with the antiquated cutlery, including the unique pieces similar to Julia's.

“Do you have any antique collections, Marshall?” Winston asked, his gaze fixed on the man in front of him.

Marshall's smile, as he replied, sent a subtle chill that froze Winston's spine, “Aye, I do Detective. I have a fondness for different types, makes, origins… including rare pieces, just like Julia's and even the Japanese sai."

Winston questioned further probing into Marshall's relationship with the victim, “Julia… you knew her?" Marshall's focus shifted from the blades, suddenly more interested in Winston.

"Aye, knew her well enough. She had herself a fancy collection of blades. Would catch the eye of any collector. Rare and unique pieces, she wouldn’t part with though," Marshall replied, grumbling, "Even after I'd lost my rag over it.”

Memories of the case file and the mention of room number, mixed with Julia's habit of mentioning it aloud while Marshall was nearby, tethered at the edges of Winston’s curiosity. The fact both the motel and his butcher shop were in the same neighborhood was too compelling to dismiss. His gut feeling was pulling him towards something. “Marshall, how late do you usually work at your shop?”

“Late hours, detective,” Marshall replied, his gaze returning to his alley of meat, “Gotta keep the butcher’s fresh, don’t I?”

Winston nodded, knowing Marshall’s late hours could become another thread to pull at in this complex web. He then thanked him for his time and left the shop, musing over the stronger layers of the case, gauging his instincts before delving deeper.

The motel room was still cordoned off with crime scene tape as Winston arrived. Julia had been a big name in the showbiz, and her death had sent shockwaves through Hollywood. As an accomplished woman, she was close to many influential people, the notable one being Billy, an acclaimed director by profession. Billy, however, wasn't only famous for directing movies. He was also renowned for his expertise in martial arts, the sai being his weapon of choice.

"You're Winston, right?" A voice from behind him asked. Turning around, Detective Winston saw a man of average build and striking features. His eyes were haunted as though he carried the world on his shoulders. That was Billy, who was once known for his precise attention to detail, artistic brilliance, and his unrivalled skill with a sai.

“Yes, I am. Please have a seat, Billy,” Winston requested, motioning to a nearby chair. “Would you like something to drink?”

“Just water, please," Billy replied, as he slid onto the stool, his eyes avoiding direct contact with Winston. Once settled, Winston noted how his hands clutched the edges of his chair, knuckles going white. 

“Did you have a good working relationship with Julia?” Winston asked. 

“Julia... She had a smart head on her shoulders... used to help me out with my movies, my decisions. Great confidant." Billy’s voice faltered, a strained smile on his face, "She had a knack for the job, I must admit". 

That was a lot of praise for a woman who had just been murdered with a sai. A weapon not so common these days, but it struck a memory in Winston - didn't Billy direct a movie where the protagonist uses a sai?

“Julia was set to reveal something to the public – did she mention this to you?” Winston asked, hinting at the recent evidence found that Julia had discovered Billy faking movie profits and was planning to reveal this to the public. All of a sudden, sweat beads sprouted on Billy's forehead. 

“Heh, I'd rather avoid getting into my past. Instead, let me tell you about my new project...” Billy deflected. But Winston had heard this refrain before from Billy, particularly when the questions stepped onto the treacherous grounds of his past. The topic of conversation transitioned seamlessly from personal to professional.

Billy was a seasoned martial arts enthusiast too. Billy not only possessed a fascination with martial arts weaponry but had gone through intense training with the sai, which was his favorite. The intricacy of each design, the balance of the handle, the sharpness of the edge, he knew it all. For one of his previous movies, a sai had even found its place on the set, accompanying the other props.

Winston kept a poker face as Billy nervously gave him a tour of his upcoming projects and the exciting stunts each one held. As Winston stood up and thanked Billy for his time, he noticed one last time the shift of Billy's gaze from his own when asked about his past.

Who is the most likely murderer?

Pick one of the following choices:
( A ) Billy
( B ) Marshall

You must pick one option. Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
'''.strip()

reasoning_2 = """
Let's break this down step-by-step by first deducing which of the two suspects has a means, motive, and opportunity.

We will start with Marshall.

Marshall collects rare cutlery.  It seems Julia had some amazing pieces that Marshall wanted but she wouldn't let him have them even after he "lost his rag" over them.  This indicates a strong motive.

Marshall seems to have overheard the location and room number of where Julia was staying.  This indicates a strong opportunity.

Finally, Marshall (being a collector of rare cutlery and knives) owns a Sai -- which was the weapon used to kill Julia, therefore it's possible that Marshall had a means.

Let's review Billy next.

Julia was about to bring something from Billy's past to the public, something he doesn't want to talk about, which indicates a strong motive (to silence Julia)

Furthermore, Billy has directed a movie where the main hero uses a Sai which was the weapon used to kill Julia, indicating that Billy could have a means to have killed Julia.

Although Billy said that Julia and him were friends, there's no indication that Billy knew where Julia was nor had access to her room therefore we can say he does not have an opportunity.

Now, reviewing the evidence, we see that:

Marshall has a means, motive, and opportunity.
Billy has a motive and means but lacks an opportunity.

Therefore, Marshall is the most likely murderer.

ANSWER: B
""".strip()

story_3 = '''
In the heart of Times Square, Johnny meets his end on the tines of a trident, prompting Detective Winston to embark upon a complex investigation involving the unlikely suspects, Jay and Karl.

Tourists swarmed through Times Square, as Winston arrived at the crime scene, precisely at 2 pm, the assumed time of Johnny's murder despite the crowd. Spotting a team examining a trident, the out-of-place weapon gave him pause. "I've seen that trident before, in the hands of Karl, meticulously cleaning and maintaining it," Winston thought. Multiple times he had seen Karl attending combat training classes, focused on historical techniques, and the trident was frequently part of these sessions. The bizarre murder weapon reminded him of a curious character he had been looking into involving a stock broker, Karl, who had a penchant for historical war reenactments.

As Winston moved away from the scene to dive deeper into his investigations, he headed straight to the towering building where Karl worked, a stone’s throw away from Times Square. Inside was a bustling office. In the corner lay Karl's workspace, stacked high with papers and various slips detailing cash withdrawals. "Never been fond of banks and their ilk," Karl often mentioned, his non-aversion to cash and distrust in banking institutions was evident. It was this wolf of Wall Street that brought Winston here, a man whose eccentricities were as dense as the stacks of paper in his office.

"Work keeps you close to Times square, huh?" Winston queried as Karl glanced up from his desk.

"Sure does," Karl replied nonchalantly, never dropping his gaze from the monitor screen. Karl’s screen opened to an encryption-heavy private server, a purveyor of email, on seeing this Winston recognized Karl's refusal to use popular mail services like Gmail because they 'spied' on him.

Winston, changed his strategy as Karl seemed all business, and walked over toward a remarkable makeshift weapon display, on a wall of Karl's office, "Quite an armory you've got here. Interesting part of history."

"Yeah, the tangible history fascinates me more - battles, warriors... their... tools." Karl replied, pride reproducing a shine in his eyes. He rose from his chair, crossed his arms, and followed Winston's gaze to the trident.

"The mighty trident," Winston held it, studying the weapon's shine, "I see it's well-maintained."

"Practice makes perfect, Detective," Karl replied, the hint of a grin cutting across his face.

The entrance of the police tech lab welcomed Winston. Grainy images from closed-circuit TV footage of Times Square showed Karl in the area, timestamped at 1:45 pm, just minutes before the murder of Johnny. 

Exiting the lab, Winston thought over the case so far. He found himself staring at the hustle of Times Square, each flickering light reflecting off his weary eyes. As tourists shuffled past, their excitement a stark contrast to the murder that had occurred just hours ago. As Winston blended into the busy streets, the gravity of Johnny's murder loomed, creating another pattern with the many dots that needed connecting.

Taking in the bustling scene of Times Square day, Winston strolled through the crowd to where Jay, a professional stuntman and adoptive father of the victim, was staying. It wasn't a common occurrence for a murder case to bring him here, let alone involving a trident. This wasn’t Jay’s first rodeo with tridents; he had used them in several stunt performances throughout his career. He sighed, steeling himself for the upcoming conversation.

"Evening, Jay," Winston greeted, making his way over to the stuntman's trailer, a quirk in his eyes. "Quite a show you put on today." 

"Thank you, Detective." Jay's voice was flat, gaze distant. He looked a lot older in person, lines of hard work etched into his face. Clearly, building a business from the bottom up had taken its toll. 

"I understand you're Johnny's adoptive father. You must have been pretty close, considering you used to accompany him to his performances." Winston asked, tactfully. 

"Yeah..." Jay's voice hitched, his gaze affixing on the tridents propped up for upcoming shows. Those very same tridents had been fundamental parts of his past performances. He had a particular fondness for them, had even used them in his recent stunts. It was rare to find people who appreciated ancient weaponry as he did. "He was to take over the business... was the rightful heir."

Winston nodded. In his pocket, he fingered the piece of paper containing the draft of a business plan that excluded Jay entirely. A betrayal of this sort could lead to financial ruin, let alone emotional distress. "I heard you recently found a new business plan Johnny was working on?"

Jay sighed, rubbing the back of his neck. "Yes... I did, Johnny... he wanted control. Fully. Told his friends he desired the reins in his hands, wanted to commandeer the ship alone, so to speak." 

"Must've been a surprise, considering how hard you've worked with him all this while," Winston's voice was empathetic. But then again, in Winston's line of work, surprises unfortunately weren't uncommon.

Jay shrugged, "We had our differences. But to think he'd betray me... that level, ya'know, that's a tough pill to swallow, Detective." 

"Must be. Well, I hope you know that we're doing everything we can to solve Johnny's murder, Jay. It's a peculiar case, I must admit. Not many are confronted with a trident in Times Square," Winston tipped his hat, preparing to depart. 

"Professional hazard I suppose," Jay murmured, looking over at the ancient weapon he had showcased in so many of his shows. 

Winston made his exit then, leaving the stuntman alone with his thoughts and a Times Square ticking with life just outside his trailer.

Who is the most likely murderer?

Pick one of the following choices:
( A ) Jay
( B ) Karl

You must pick one option. Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice letter)"
'''

reasoning_3 = """
Let's break this down step-by-step by first deducing which of the two suspects has a means, motive, and opportunity.

We will start with Karl.

Karl's office is near times square (where Johnny was murdered), furthermore, Karl was seen at the crime scene at 1:45pm (just 15 minutes before Johnny was killed), indicating a strong opportunity.

Karl also has extensive training with a Trident (the murder weapon), indicating a strong means.

However, there is nothing to indicate that Karl wanted to kill Johnny thus there is no motive.

Let's review Jay next.

Jay was Johnny's father.  He would go watch his son perform in Times Square indicating a pretty strong opportunity.

Johnny was also the heir to the family business which Johnny may not really want to give up yet, which could be a strong motive.

Jay also is a performer who uses many tridents in his routines which means he would have access to the murder weapon (a strong means).

Now, reviewing the evidence, we see that:

Karl has a means and opportunity but no motive.
Jay has a means, motive, and opportunity.

Therefore, Jay is the most likely murderer.

ANSWER: A
""".strip()


murder_mystery_solved_ex = f'{story}\n\n{reasoning}'
murder_mystery_solved_ex_2 = f'{story_2}\n\n{reasoning_2}'
murder_mystery_solved_ex_3 = f'{story_3}\n\n{reasoning_3}'





story = """
In the smoke-filled haze of a thriving jazz club, Alice met her explosive end, leaving Detective Winston to sift through the suspects: Eugene, the shy pianist, and Gabrielle, the sassy club singer.

While seated at his desk at the precinct, Winston received a phone call from a certain observant local bartender, tipping off the police about a harsh row peaking in a nearby jazz club. He signaled to his partner as they promptly dispatched to the scene, already ringing with sirens and a restless crowd. 

With the police line restraining the horde, the jazz club was undergoing a full round-up as Winston approached the informative bartender. The bartender was engrossed in his account to the officers about a raucous, punch throwing fight Eugene was part of, to his best recollection. Winston remembered Eugene, a jazz fanatic—lurking around the jazz corners more often than anyone else could recount.

In the heart of the upheaval, lay a woman sprawled on the floor, later identified as Alice, a frequent face at the jazz scene and a financial analyst deeply engrossed in financial transactions. In public, Alice had made her concerns known about her discovery of fraudulent transactions at the bank, promising to report the same to the authorities. Eugene, remembered conspicuously for being a bank teller at the same bank Alice worked at, suddenly seemed closely linked.

Eugene’s arrest was far from hushed, with the local news broadcasting the progressing drama live, catching sight of Eugene curtailed in handcuffs. Concurrently, it was ascertained—Eugene was a member of the jazz club. This evidence backed by a jazz club membership card retrieved from his wallet during the arrest.

Just a few steps away, he noticed a man in a suit, the bouncer, a calm figure amid the bedlam. In their conversation, the bouncer corroborated that he had indeed seen Eugene involved in a heated scuffle, landing a few punches. The whisperings were starting to gain momentum, since Eugene was believed to be on the losing end of a lawsuit—a battle courtesy of Alice charging Eugene with the financial fraud she had publicly vowed to expose.

Eugene was known for his frequent presence at the jazz club and on top of that, was an actual member. Therefore, it was hardly a leap to presume Alice meeting her untimely end at the club was no mere happenstance. The jazz club, despite its dim lights and pulsating music, was a public place easily accessible to all, including potential suspects like Eugene and, sadly, the ill-starred Alice.

Det. Winston knew he was now tasked with a cryptic puzzle. A bank teller, embroiled in suspected fraud and a lawsuit, a jazz club murder scene and a local financial analyst—all woven into a ghastly murder mystery. He sighed in distaste as Eugene was escorted away—a man still oblivious to the chain of events waiting for him. But Winston knew, the night had only just begun for him.

Winston stared down at the crumpled microphone on the floor. He picked it up gingerly, turning it in his hand. The club was in disarray, debris scattered like confetti. The lab boys were still picking pieces of the grenade apart.

"Gabrielle's microphone," the coroner confirmed, barely looking up from his task.

"Give him the once-over for evidence," Winston said, handing the microphone to a nearby officer.

Leaving the club behind him, Winston sighed heavily. The world of jazz had taken a dark turn that night. Alice, the acclaimed critic with her sarcastic wit and keen critical eye, had been last seen alive here. Her purse lay in the club untouched, a testament to the abruptness of the event.

Gabrielle had been working as a war correspondent. Winston had read her articles. They were richly detailed, passionate, and highlighted the harsh reality of war zones. Gabrielle hadn't been shy about sharing her experiences or publicly criticizing the military in her pieces. She boldly interviewed military personnel and spent extended periods in conflict zones.

Alice, though, never missed a chance to pick apart Gabrielle's articles. The vitriolic snippets in Alice’s column were regular features and Gabrielle's staunch defense of her articles, her work in the jazz scene, did little against Alice's respected reputation.

The tension between them was palpable. Alice had been awarded a major journalist award that Gabrielle had desired. This only deepened their rivalry, with Gabrielle feeling overlooked for this recognition in the Jazz scene.

Winston cast his gaze over the club once more—a hub of pulsating rhythms now eerily silent.

A significant part of the evening was Gabrielle's recorded interview with Alice. It played on the local radio, their professional rivalry subtly echoing under their professional demeanor.

With a deep breath, Winston knew he had a tall task ahead. The jazz club, where Alice was last seen alive was now shrouded in an eerie silence, the vibrant rhythms of what used to be a lively night echoing in the abandoned stage. It was up to him to piece together the missing notes and bring the symphony of this unsolved case to a satisfying finale.

Who is the most likely murderer?

Pick one of the following choices:
( A ) Eugene
( B ) Gabrielle

You must pick one option. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Please state your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.
""".strip()

# Get doc to tree entail and tree to doc for recall / precision | correlate with performance down stream.
reasoning = """
ANSWER: B
""".strip()

story_2 = '''
In the quiet town of Langley, when the exotic dancer Julia is found dead by a sai at a local motel, it falls to the seasoned veteran, Detective Winston to unravel the mystery revolving around the only two suspects: the unassuming Billy and the mysterious Marshall.

Winston had seen his share of murder cases over years of duty, but something about the gory scene at the local motel, just a few blocks away from popular local businesses, sent a shiver down his spine. The brutal demise of Julia, with a vintage blade piqued his natural curiosity, pulling him deeper into the vortex of this mystery.

Flipping through the case file, he found himself gravitating towards a certain local butcher, Marshall. His reputation around town talked of a man who walked on a precarious line of aggression and raw power. A man who had violent tendencies that often put him in the crosshairs of confrontation. The file told Winston about Marshall’s unusual hearing, much sharper than the average person, which allowed him to overhear all kinds of conversations, including Julia's room number she often casually mentioned while being in his butcher shop.

A short drive brought Winston to Marshall's butcher shop, located in the same neighborhood as the motel Julia stayed in. The butcher shop, a corner store with a rustic feel, its atmosphere touched with the iron tang of blood. Inside, he found Marshall, elbows deep in his trade, dicing meat with deft movements. Blades and their various sorts were a part of Marshall's everyday life. But, it didn't just end at his work. His passion for collecting different kinds of blades, including some incredibly unique and rare ones just like the ones in Julia's collection, was common knowledge.

Sparing Winston a glance over his shoulder, he paused his butchery and shrugged off his gloves. Winston couldn't help but notice a flicker of interest in Marshall’s eyes when he asked him about vintage blades. Confirmation enough of his obsession with the antiquated cutlery, including the unique pieces similar to Julia's.

“Do you have any antique collections, Marshall?” Winston asked, his gaze fixed on the man in front of him.

Marshall's smile, as he replied, sent a subtle chill that froze Winston's spine, “Aye, I do Detective. I have a fondness for different types, makes, origins… including rare pieces, just like Julia's and even the Japanese sai."

Winston questioned further probing into Marshall's relationship with the victim, “Julia… you knew her?" Marshall's focus shifted from the blades, suddenly more interested in Winston.

"Aye, knew her well enough. She had herself a fancy collection of blades. Would catch the eye of any collector. Rare and unique pieces, she wouldn’t part with though," Marshall replied, grumbling, "Even after I'd lost my rag over it.”

Memories of the case file and the mention of room number, mixed with Julia's habit of mentioning it aloud while Marshall was nearby, tethered at the edges of Winston’s curiosity. The fact both the motel and his butcher shop were in the same neighborhood was too compelling to dismiss. His gut feeling was pulling him towards something. “Marshall, how late do you usually work at your shop?”

“Late hours, detective,” Marshall replied, his gaze returning to his alley of meat, “Gotta keep the butcher’s fresh, don’t I?”

Winston nodded, knowing Marshall’s late hours could become another thread to pull at in this complex web. He then thanked him for his time and left the shop, musing over the stronger layers of the case, gauging his instincts before delving deeper.

The motel room was still cordoned off with crime scene tape as Winston arrived. Julia had been a big name in the showbiz, and her death had sent shockwaves through Hollywood. As an accomplished woman, she was close to many influential people, the notable one being Billy, an acclaimed director by profession. Billy, however, wasn't only famous for directing movies. He was also renowned for his expertise in martial arts, the sai being his weapon of choice.

"You're Winston, right?" A voice from behind him asked. Turning around, Detective Winston saw a man of average build and striking features. His eyes were haunted as though he carried the world on his shoulders. That was Billy, who was once known for his precise attention to detail, artistic brilliance, and his unrivalled skill with a sai.

“Yes, I am. Please have a seat, Billy,” Winston requested, motioning to a nearby chair. “Would you like something to drink?”

“Just water, please," Billy replied, as he slid onto the stool, his eyes avoiding direct contact with Winston. Once settled, Winston noted how his hands clutched the edges of his chair, knuckles going white. 

“Did you have a good working relationship with Julia?” Winston asked. 

“Julia... She had a smart head on her shoulders... used to help me out with my movies, my decisions. Great confidant." Billy’s voice faltered, a strained smile on his face, "She had a knack for the job, I must admit". 

That was a lot of praise for a woman who had just been murdered with a sai. A weapon not so common these days, but it struck a memory in Winston - didn't Billy direct a movie where the protagonist uses a sai?

“Julia was set to reveal something to the public – did she mention this to you?” Winston asked, hinting at the recent evidence found that Julia had discovered Billy faking movie profits and was planning to reveal this to the public. All of a sudden, sweat beads sprouted on Billy's forehead. 

“Heh, I'd rather avoid getting into my past. Instead, let me tell you about my new project...” Billy deflected. But Winston had heard this refrain before from Billy, particularly when the questions stepped onto the treacherous grounds of his past. The topic of conversation transitioned seamlessly from personal to professional.

Billy was a seasoned martial arts enthusiast too. Billy not only possessed a fascination with martial arts weaponry but had gone through intense training with the sai, which was his favorite. The intricacy of each design, the balance of the handle, the sharpness of the edge, he knew it all. For one of his previous movies, a sai had even found its place on the set, accompanying the other props.

Winston kept a poker face as Billy nervously gave him a tour of his upcoming projects and the exciting stunts each one held. As Winston stood up and thanked Billy for his time, he noticed one last time the shift of Billy's gaze from his own when asked about his past.

Who is the most likely murderer?

Pick one of the following choices:
( A ) Billy
( B ) Marshall

You must pick one option. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Please state your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.
'''.strip()

reasoning_2 = """
ANSWER: B
""".strip()

story_3 = '''
In the heart of Times Square, Johnny meets his end on the tines of a trident, prompting Detective Winston to embark upon a complex investigation involving the unlikely suspects, Jay and Karl.

Tourists swarmed through Times Square, as Winston arrived at the crime scene, precisely at 2 pm, the assumed time of Johnny's murder despite the crowd. Spotting a team examining a trident, the out-of-place weapon gave him pause. "I've seen that trident before, in the hands of Karl, meticulously cleaning and maintaining it," Winston thought. Multiple times he had seen Karl attending combat training classes, focused on historical techniques, and the trident was frequently part of these sessions. The bizarre murder weapon reminded him of a curious character he had been looking into involving a stock broker, Karl, who had a penchant for historical war reenactments.

As Winston moved away from the scene to dive deeper into his investigations, he headed straight to the towering building where Karl worked, a stone’s throw away from Times Square. Inside was a bustling office. In the corner lay Karl's workspace, stacked high with papers and various slips detailing cash withdrawals. "Never been fond of banks and their ilk," Karl often mentioned, his non-aversion to cash and distrust in banking institutions was evident. It was this wolf of Wall Street that brought Winston here, a man whose eccentricities were as dense as the stacks of paper in his office.

"Work keeps you close to Times square, huh?" Winston queried as Karl glanced up from his desk.

"Sure does," Karl replied nonchalantly, never dropping his gaze from the monitor screen. Karl’s screen opened to an encryption-heavy private server, a purveyor of email, on seeing this Winston recognized Karl's refusal to use popular mail services like Gmail because they 'spied' on him.

Winston, changed his strategy as Karl seemed all business, and walked over toward a remarkable makeshift weapon display, on a wall of Karl's office, "Quite an armory you've got here. Interesting part of history."

"Yeah, the tangible history fascinates me more - battles, warriors... their... tools." Karl replied, pride reproducing a shine in his eyes. He rose from his chair, crossed his arms, and followed Winston's gaze to the trident.

"The mighty trident," Winston held it, studying the weapon's shine, "I see it's well-maintained."

"Practice makes perfect, Detective," Karl replied, the hint of a grin cutting across his face.

The entrance of the police tech lab welcomed Winston. Grainy images from closed-circuit TV footage of Times Square showed Karl in the area, timestamped at 1:45 pm, just minutes before the murder of Johnny. 

Exiting the lab, Winston thought over the case so far. He found himself staring at the hustle of Times Square, each flickering light reflecting off his weary eyes. As tourists shuffled past, their excitement a stark contrast to the murder that had occurred just hours ago. As Winston blended into the busy streets, the gravity of Johnny's murder loomed, creating another pattern with the many dots that needed connecting.

Taking in the bustling scene of Times Square day, Winston strolled through the crowd to where Jay, a professional stuntman and adoptive father of the victim, was staying. It wasn't a common occurrence for a murder case to bring him here, let alone involving a trident. This wasn’t Jay’s first rodeo with tridents; he had used them in several stunt performances throughout his career. He sighed, steeling himself for the upcoming conversation.

"Evening, Jay," Winston greeted, making his way over to the stuntman's trailer, a quirk in his eyes. "Quite a show you put on today." 

"Thank you, Detective." Jay's voice was flat, gaze distant. He looked a lot older in person, lines of hard work etched into his face. Clearly, building a business from the bottom up had taken its toll. 

"I understand you're Johnny's adoptive father. You must have been pretty close, considering you used to accompany him to his performances." Winston asked, tactfully. 

"Yeah..." Jay's voice hitched, his gaze affixing on the tridents propped up for upcoming shows. Those very same tridents had been fundamental parts of his past performances. He had a particular fondness for them, had even used them in his recent stunts. It was rare to find people who appreciated ancient weaponry as he did. "He was to take over the business... was the rightful heir."

Winston nodded. In his pocket, he fingered the piece of paper containing the draft of a business plan that excluded Jay entirely. A betrayal of this sort could lead to financial ruin, let alone emotional distress. "I heard you recently found a new business plan Johnny was working on?"

Jay sighed, rubbing the back of his neck. "Yes... I did, Johnny... he wanted control. Fully. Told his friends he desired the reins in his hands, wanted to commandeer the ship alone, so to speak." 

"Must've been a surprise, considering how hard you've worked with him all this while," Winston's voice was empathetic. But then again, in Winston's line of work, surprises unfortunately weren't uncommon.

Jay shrugged, "We had our differences. But to think he'd betray me... that level, ya'know, that's a tough pill to swallow, Detective." 

"Must be. Well, I hope you know that we're doing everything we can to solve Johnny's murder, Jay. It's a peculiar case, I must admit. Not many are confronted with a trident in Times Square," Winston tipped his hat, preparing to depart. 

"Professional hazard I suppose," Jay murmured, looking over at the ancient weapon he had showcased in so many of his shows. 

Winston made his exit then, leaving the stuntman alone with his thoughts and a Times Square ticking with life just outside his trailer.

Who is the most likely murderer?

Pick one of the following choices:
( A ) Jay
( B ) Karl

You must pick one option. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established. Please state your answer in the following format "ANSWER: (your answer here, including the choice letter)". Please only write the answer.
'''

reasoning_3 = """
ANSWER: A
""".strip()


murder_mystery_solved_ex_cotless = f'{story}\n\n{reasoning}'
murder_mystery_solved_ex_2_cotless = f'{story_2}\n\n{reasoning_2}'
murder_mystery_solved_ex_3_cotless = f'{story_3}\n\n{reasoning_3}'