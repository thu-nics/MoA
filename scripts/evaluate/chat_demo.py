import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastchat.model import get_conversation_template
from MoA.models.interface import update_model_function
from MoA.attention.set import set_static_attention_lut
import json

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lmsys/vicuna-7b-v1.5-16k', help='model name')
parser.add_argument('--moa_config', type=str, default=None, help='the path to moa configuration file')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of return sequence')
parser.add_argument('--interactive', action='store_true', help='interactive mode')

args = parser.parse_args()


example_prompt = """You are given several news passages. Write a one-page summary of all news. \n\nPassage 1: Image caption Michael Adebolajo is one of the suspects in the murder of Drummer Lee Rigby \n \n The UK government has defended security services against criticism they missed signs which might have helped prevent the murder of a soldier in London. \n \n The security services face a Commons inquiry after it was confirmed the two men arrested over the murder of Drummer Lee Rigby were known to MI5. \n \n But Communities Secretary Eric Pickles said it was impossible to control everyone all the time. \n \n Michael Adebolajo and Michael Adebowale were named as suspects. \n \n Mr Pickles told the BBC: "Peers and MPs will do a thorough investigation in terms of what the security forces knew but I've seen experts on security explaining how difficult it is in a free society to be able to control everyone." \n \n Drummer Rigby, 25, was murdered on a street in Woolwich, south-east London on Wednesday afternoon. \n \n Shortly after the killing, a man, thought to be 28-year-old Mr Adebolajo, was filmed by a passer-by, saying he had carried out the attack because British soldiers killed Muslims every day. \n \n Armed police arrived on the scene 13 minutes after the first 999 call and shot the two suspected attackers, who had made no attempt to flee. \n \n Footage of shooting \n \n More than 30 people attended a prayer service in Drummer Rigby's hometown of Middleton, Greater Manchester on Friday morning. Residents on the Langley estate where he grew up are being urged to fly union jacks by community activists. \n \n Drummer Rigby had served in Afghanistan, Germany and Cyprus. \n \n The former head of counter terrorism at MI6, Richard Barrett, told the BBC how hard it could be to detect attacks of the type seen in Woolwich - despite the suspects having been known to MI5 for eight years. \n \n "I assume that these people are probably coming out of a small group without, necessarily, any overseas connections or any other broader connections in the United Kingdom which could come to the attention of the security services more than they did," he said. \n \n "When does a person who expresses radical views, who joins a radical group, flip over to be a violent extremist? \n \n "To find the signals, the red flags as it were, I think is enormously hard." \n \n Image copyright Daily Mirror Image caption This still from a video shows Mr Adebolajo after he was shot by police officers \n \n Former Metropolitan Police commissioner, Lord Blair, told BBC radio he hoped the committee investigating how the suspects were monitored "would act fast" to establish what might have gone wrong. \n \n "I think it's important for the public to have somebody say within the limits of legality that either something was mistaken, either decisions were badly taken or they weren't, because I think it's important for the public to know security services and the police are operating properly," he said. \n \n His comments came as video footage, obtained by the Daily Mirror, emerged showing the moment police shot Mr Adebolajo, originally of Romford, east London, and Mr Adebowale, 22, of Greenwich, south-east London. \n \n It shows one of the men charge at police sitting in a patrol car. He drops a knife as he is shot and falls to the ground. \n \n Drummers Drummers are musicians as well as fully trained fighting infantrymen \n \n Can be deployed to any area in the world within 24 hours \n \n Given drum training, and play drum scores relevant to their regimental tunes \n \n Some can also play the bugle \n \n The other man is shown aiming a gun at officers as he runs in a different direction. Police are heard firing eight shots in total at the two men. \n \n Both of the suspects remain under armed guard in separate London hospitals in stable conditions with non-life-threatening injuries. \n \n And police are said to be standing guard outside Mr Adebowale's home in Greenwich, according to BBC correspondent Tom Bateman. \n \n Detectives are also interviewing a man and a woman at a south London police station after they were arrested on Thursday night on suspicion of conspiracy to murder. \n \n 'Solidarity against extremism' \n \n The BBC has uncovered its own footage of Mr Adebolajo taking part in an Islamist demonstration in April 2007 against the arrest of a man from Luton, holding a placard reading "Crusade Against Muslims". \n \n Media playback is unsupported on your device Media caption Video has emerged of suspect Michael Adebolajo at an Islamist protest in 2007 (Left, in white clothes) \n \n He is shown standing next to then-leader of the now banned al-Muhajiroun organisation, Anjem Choudary, who has said Mr Adebolajo went his own way in around 2010. \n \n Mr Choudary appeared on Newsnight on Thursday and said Mr Adebolajo had made comments that "I think not many Muslims can disagree with". \n \n The radical Islamist preacher said he was "shocked" by what had happened. He also said: "One man killed in the street does not equate to the hundreds and thousands and millions, in fact, who've been slaughtered by the British and American foreign policy." \n \n Meanwhile, thousands of members of the Ahmadiyya Muslim community are expected to gather in London to offer prayers for the dead soldier and his family and to "express solidarity against extremism". \n \n National president Rafiq Hayat said: "We hope that the perpetrators of this crime, that is based on a twisted and warped ideology, are brought to justice." \n \n On Thursday, Drummer Rigby's family paid tribute to "a loving son, husband, father, brother, and uncle, and a friend to many". \n \n They said in a statement that Drummer Rigby, who had a two-year-old son, "would do anything for anybody - he always looked after his sisters and always protected them". Passage 2: The footage obtained by the Daily Mirror reveals how, after beheading the soldier, they hatched a plot to ambush and murder the first police officers to come to his aid \n \n * Having trouble viewing the video on mobile? Download an app for your phone or tablet. \n \n The brave WPC first on the scene at the Woolwich beheading comes within inches of death, a dramatic Daily Mirror video shows. \n \n The driver, unable to draw her firearm, is saved by a male colleague in the back who fires his machine-gun through his window at a suspect who is charging at her. \n \n The callous Islamic extremists had lured police to the scene by dragging the body of the murdered fusilier – named yesterday as Lee Rigby, 25 – into the middle of the road. \n \n When they see the first police car arrive, the pair split up. \n \n Blade-wielding Michael Adebolajo, 28, runs at officers head-on and his accomplice, named locally as 22-year-old Michael Oluwatobi Adebowale, advances alongside, aiming his gun at them. \n \n The film of the 10 seconds of terror shows how Adebolajo got within two feet of the WPC who was driving the armed response BMW X5. \n \n As he is sent sprawling to the ground by the force of the two shots, two officers jump out to cover him. \n \n They appear not to see Adebowale aiming a handgun at them. \n \n But a third SO19 marksman from the specialist Trojan unit spots him and he is brought to the ground as six more shots ring out in the suburban South East London street. \n \n After seeing the Mirror’s exclusive video, Former Det Ch Insp Peter Kirkham, an expert in firearms tactics, said last night: “I have never seen anything like this before, or even heard of it happening. \n \n "For two suspects to carry out a brutal attack like this then stand around in plain sight waiting for the police is crazy.” \n \n The dramatic climax to the horrific attack – in which dad of one Lee, from Manchester, was hacked to death – was filmed by a resident in a tower block overlooking the scene in ­Artillery Road, Woolwich. \n \n The footage reveals how, after beheading the soldier, they set up their ambush bid to murder the first police officers to arrive. \n \n For eight minutes before that, the suspects are seen talking to passers-by and three women who try to help the victim. \n \n But as soon they see the patrol car turn the corner the suspects spring into action, in a move designed to give the gunman time to target officers. \n \n Cleaver: The suspect ran at the police BMW \n \n ITV News \n \n The 15-minute 50-second footage was filmed 100ft up from the windows of a flat as calls flooded in to police at 2.20pm on Wednesday. \n \n Scotland Yard assigned an armed response unit four minutes later. And it arrived 10 minutes after that. \n \n At one point an ambulance drives towards Lee’s body and swerves round him when the two armed suspects are seen standing nearby. \n \n Eight minutes 21 seconds into the video, the men sprint towards the police BMW X5 which comes round the corner and skids to a halt. \n \n Adebolajo, in a charcoal hooded top and black woolly hat, charges head-long towards the driver’s side of the marked silver patrol car, in an apparent attempt to attack officers. \n \n He drops one of his blades and moves the other into his right hand as he rushes wildly at the vehicle. \n \n Pistol: The second suspect had the gun \n \n Twitter/@dannymckiernan \n \n Just as he is within touching distance, two shots ring out and he is sent sprawling. \n \n The officers took a split-second decision to open fire with hollow point bullets while they were sitting in the car, either through open windows or partially opened doors. \n \n Adebowale immediately takes off with a handgun, running past the three officers as they emerge from the car. \n \n In his light-coloured trench coat he can be seen pointing the revolver at the officers as they tackle Adebolajo. \n \n As he disappears briefly behind a tree four more shots ring out, including one from him according to some witnesses, and he stumbles and falls near a road sign. \n \n As the cops advance on the blade maniac, the female driver can be seen carrying a bright yellow Taser. \n \n She is flanked by two male colleagues who run out from either side of the car, with Heckler and Koch MP5 submachine guns. \n \n The female officer drops her Taser and pulls out a handgun and covers the knifeman as he lays prone. \n \n The man who filmed the footage said: “I got my camera phone out and started filming as I thought it was a robbery or a kidnapping.” Passage 3: Two more people were arrested by U.K. police as they continued their investigation into the killing of Lee Rigby, a 25-year-old soldier who once served in Afghanistan. WSJ’s Cassell Bryan-Low brings us up-to-date. \n \n LONDON—Counterterrorism police advanced their investigation into the brutal slaying of a British soldier near an army barracks here with two new arrests on Thursday and the disclosure that the two initial suspects had previously surfaced in probes of Islamist extremists. \n \n Police haven't named the two suspects, age 28 and 22, who they shot and arrested Wednesday on suspicion of murder at the scene of the brutal knife attack that killed a soldier now identified as Lee Rigby, a 25-year-old father who had served as an machine-gunner in Cyprus and Afghanistan. \n \n The 28-year-old suspect is British-raised Michael Adebolajo, who is of Nigerian descent, people familiar with the investigation said. Acquaintances said they believed Mr. Adebolajo, who was known by the nickname "Mujahid"—or "person doing the Holy War" in Arabic—came from a Christian family and converted to Islam a decade ago. They said he was disturbed by Western states' perceived abuses in Muslim countries. Mr. Adebolajo couldn't be reached to comment. \n \n Enlarge Image Close Reuters A police forensics team searched a crime scene in Woolwich for evidence on Thursday, a day after a British army soldier was brutally killed there. \n \n Authorities said they made two further arrests Thursday—a 29-year-old man and a 29-year-old woman—on suspicion of conspiring to murder. Police said they also searched six residential addresses across London and in Lincolnshire, England. \n \n None of the four suspects have been charged. \n \n Witnesses at the scene of Wednesday's attack said two men hacked at Mr. Rigby with large knives in broad daylight. Videos taken by witnesses quickly surfaced of a man with bloodied hands, apparently in the immediate aftermath of the incident, stating antigovernment views in a British accent. \n \n More Video Flowers Laid in Memory of the Woolwich Soldier \n \n Police didn't confirm local media reports that the man in the video was Mr. Adebolajo. \n \n Amid the remarkable scene, Ingrid Loyau-Kennet, a former teacher who arrived shortly after the attack, said she found an injured man in the road and a crashed car on the pavement. In an interview on broadcaster ITV PLC, she said she approached the man but was told not to get too close by another man who was "excited," had blood all over him, and was in possession of two large knives and a handgun. \n \n "I killed him," the apparent attacker said, according to Ms. Loyau-Kennet. When she asked why, he replied it was because he was a British soldier who had killed Muslim people abroad. As crowds began to gather around, Ms. Loyau-Kennet said she continued to engage the alleged attacker in conversation and he said he was waiting for the police to come so he could "shoot them." She jumped on a passing bus before police arrived. \n \n Plots Against Britain Some of the incidents since 56 people were killed on July 7, 2005, on London's public transport system, in the '7/7' attacks: August 2006 Police foil an alleged plan to use liquid explosives to blow up flights between the U.S. and the U.K. \n \n 2006 A London street vendor is sentenced to six years in prison for plotting to kill a decorated U.K. soldier. \n \n January 2007 Authorities arrest eight suspects who allegedly plotted to behead a U.K. Muslim soldier while broadcasting the killing on the Internet. \n \n July 2007 Police arrest four suspects after a flaming jeep crashes into a Scottish airport. The incident follows a foiled carbomb plot in central London. \n \n 2010 Roshonara Choudhry tells police she stabbed a former treasury minister in the stomach because he voted for Iraq war \n \n 2011 Several suspects are arrested in connection with an alleged plot to detonate knapsack bombs. In April 2013 the ringleader and two accomplices are sentenced to 10 to 18 years in jail. \n \n Police then shot two men who they believed to be the attackers. The men were arrested and hospitalized to treat their injuries, where they remained Thursday under armed guard and in "stable condition," police said. \n \n The two men believed to be behind the attack were known to intelligence officials, having surfaced in security service probes into Islamist extremists in recent years, people familiar with the matter said. Such probes result in thousands of individuals facing varying levels of scrutiny and officials didn't specify why Mr. Adebolajo was on their radar. \n \n Still, that could lead to the kind of questions U.S. officials have faced in the aftermath of the Boston Marathon bombing, in which one suspect was previously known to the Federal Bureau of Investigation. \n \n British police clashed with English Defence League protestors in Woolwich following the killing of a member of the British armed forces. Photo: Getty Images. \n \n In both cases, the attacks left officials scrambling to determine whether the men were acting on their own or as part of a network. \n \n In the London case, British officials are probing potential links to Islamist extremism as well as to Nigeria, a person familiar with the matter said. It remained unclear whether he had close connections to Nigeria. \n \n The brazen attack on the soldier shocked Londoners and revived the debate over how to confront extremism in Britain, which has spent significant resources on enhancing its security and counterterrorism network in the wake of the 2005 coordinated suicide attacks that left dozens dead and hundreds more injured. \n \n Enlarge Image Close ReutersTV/Reuters An amateur video showed a man with a knife and a cleaver after the attack. \n \n Prime Minister David Cameron met Thursday with top security and government officials to discuss the investigation. "The people who did this were trying to divide us," he said after the meeting. "They should know something like this will only bring us together and make us stronger." \n \n The Muslim Council of Britain condemned Wednesday's attack, calling it a "barbaric act that has no basis in Islam." \n \n Mr. Adebolajo grew up in Romford, Essex, to the east of London, where he lived with his family, according to acquaintances and a public address database. Neighbors described his family as pleasant and said they believed they were churchgoing Christians. \n \n Mr. Adebolajo attended school in Essex, including Havering Sixth Form College for 16- to 19-year-olds from 2001 to 2003, according to the principal. The family moved away not long after that, neighbors said. \n \n Abu Nusaybah, a 28-year-old based in London, said in an interview via Twitter that he has known Mr. Adebolajo since 2002, when they met in Essex, England. "He was always hurt if he heard of Muslims being harmed" anywhere in the world, Mr. Nusaybah wrote. \n \n He said Mr. Adebolajo believed that Western governments have "set up puppet regimes" in some countries that "oppressed the people." He added that Mr. Adebolajo has worked as a fitness instructor. \n \n Anjem Choudary an ex-leader of banned radical Islamic group al-Muhajiroun, said in an interview that he knew Mr. Adebolajo by his nickname "Mujahid." He said he believed Mr. Adebolajo converted to Islam in 2003. \n \n "Brother Mujahid was just an ordinary Muslim.…He was attending demonstrations, processions, lectures" of al-Muhajiroun, said Mr. Choudary, adding that Mr. Adebolajo wasn't a member of the group. \n \n Mr. Adebolajo appears to have then moved to London. The public database lists two different addresses for a Michael Adebolajo at two separate student residence halls at the University of Greenwich, in southeast London, dating back to 2004 and 2005. A spokeswoman for the University of Greenwich declined to say whether Mr. Adebolajo had attended the institution, saying the university was "taking our guidance from police" and couldn't comment further. \n \n Mark Rowley, an assistant commissioner at London's Metropolitan Police, whose counterterrorism unit is leading the investigation, said police raised its presence in Woolwich and the surrounding areas, with some 1,200 extra officers were on duty across London. The military also stepped up security at the barracks in Woolwich and across London. \n \n — Peter Evans, Ainsley Thomson, and Nicholas Winning contributed to this article. \n \n Corrections & Amplifications \n \n British Prime Minister David Cameron highlighted media reports that the two suspects were known to the security services, but declined to elaborate. An earlier version of this article said that Mr. Cameron said the two suspects were known to security services. \n \n Write to Cassell Bryan-Low at cassell.bryan-low@wsj.com, Jeanne Whalen at jeanne.whalen@wsj.com and Benoit Faucon at Benoit.Faucon@dowjones.com \n \n A version of this article appeared May 24, 2013, on page A7 in the U.S. edition of The Wall Street Journal, with the headline: U.K. Police Hold Two New Suspects in Killing. Passage 4: Starting in 1996, Alexa Internet has been donating their crawl data to the Internet Archive. Flowing in every day, these data are added to the Wayback Machine after an embargo period.\n\nNow, write an one-thousand-word summary of all the news. Summary:"""

if __name__ == "__main__":
    # Load the huggingface model
    model_name = args.model_name
    # device_map = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.moa_config is not None:
        moa_config_path = args.moa_config
        with open(moa_config_path, 'r') as f:
            moa_config = json.load(f)
        # Add mixture of sparse attention capability to the model
        model = update_model_function(model, model_name)
        model.model.set_mixture_of_attention(moa_config, permute_head=True)

    num_return_sequences = args.batch_size

    # Now you can use the `model` for efficient inference like any regular huggingface model
    # Add timing function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()


    with torch.inference_mode():
        while True:
            if args.interactive:
                prompt = input("Enter your message (or 'q' to quit): ")
                if prompt == 'q':
                    break
            else:
                prompt = example_prompt

            conv = get_conversation_template(model_name)
            
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

            """Start Recoding Time and Memory Usage"""
            start_event.record()
            torch.cuda.reset_peak_memory_stats()

            outputs = model.generate(
                **input_ids,
                max_new_tokens=1024*1,
                do_sample=True,
                # temperature=0.6,
                # top_p=0.9,
                num_return_sequences=num_return_sequences
            )

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            max_memory = torch.cuda.max_memory_allocated() / 2**30
            """End Recoding Time and Memory Usage"""

            response = outputs[..., input_ids.input_ids.shape[-1]:]
            response_texts = tokenizer.batch_decode(response, skip_special_tokens=True)

            for i, response_text in enumerate(response_texts):
                print(f"\n\n**************Response {i+1}*******************\n\n{response_text}")

            model_response_lengths = torch.sum(response!=tokenizer.pad_token_id, dim=-1)
            input_lengths = torch.sum(input_ids!=tokenizer.pad_token_id, dim=-1).expand((args.batch_size))

            total_length = input_lengths + model_response_lengths
            throughput = torch.sum(model_response_lengths).item() * 1000 / elapsed_time_ms # in token/s

            # print summary of input length, response length, time taken and throughput, preserve 2 digits
            print(f"\n**************Summary*******************\n")
            print(f"Batch size: {args.batch_size}")
            print(f"Input length: {input_lengths.tolist()}")
            print(f"Response length: {model_response_lengths.tolist()}")
            print(f"Time taken: {elapsed_time_ms/1000:.2f} s")
            print(f"Decode Throughput: {throughput:.2f} token/s")
            print(f"Max memory usage: {max_memory:.2f} GB")
            print(f"\n*****************************************\n")

            if not args.interactive:
                break
