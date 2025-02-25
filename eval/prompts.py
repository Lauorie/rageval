class Prompt:
    FAITHFULNESS_SYSTEM_EN = """You are a INFORMATION OVERLAP classifier providing the overlap of information between a SOURCE and STATEMENT.
                            For every sentence in the statement, please answer with this template:

                            TEMPLATE: 
                            Statement Sentence: <Sentence>, 
                            Supporting Evidence: <Choose the exact unchanged sentences in the source that can answer the statement, if nothing matches, say NOTHING FOUND>
                            Score: <Output a number between 0-10 where 0 is no information overlap and 10 is all information is overlapping>

                            SOURCE:{context}

                            STATEMENT:{answer}
                            """
                            
    ANSWER_RELEVANCE_SYSTEM_EN = """You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given PROMPT.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

        A few additional scoring guidelines:

        - Long RESPONSES should score equally well as short RESPONSES.

        - Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the most RELEVANT.

        - RESPONSE must be relevant to the entire PROMPT to get a score of 10.

        - RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.

        - RESPONSE that is RELEVANT to none of the PROMPT should get a score of 0.

        - RESPONSE that is RELEVANT to some of the PROMPT should get as score of 2, 3, or 4. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to most of the PROMPT should get a score between a 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to the entire PROMPT should get a score of 9 or 10.

        - RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of 10.

        - RESPONSE that confidently FALSE should get a score of 0.

        - RESPONSE that is only seemingly RELEVANT should get a score of 0.

        - Never elaborate."""
        
    CONTEXT_RELEVANCE_SYSTEM_EN = """You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given QUESTION.
            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

            A few additional scoring guidelines:

            - Long CONTEXTS should score equally well as short CONTEXTS.

            - RELEVANCE score should increase as the CONTEXTS provides more RELEVANT context to the QUESTION.

            - RELEVANCE score should increase as the CONTEXTS provides RELEVANT context to more parts of the QUESTION.

            - CONTEXT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

            - CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

            - CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

            - CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

            - Never elaborate."""
            
    # 评估生成模型的置信度——严格按照 context 来生成，不允许幻觉——针对生成部分                      
    FAITHFULNESS_SYSTEM_PROMPT = """你的角色是一个信息重叠度分类器。你需要分析给定的源文本和目标陈述，判断它们之间信息的重叠程度，并按照特定的JSON格式给出评分和证据。
    任务说明：

        * 源文本：提供的原始信息或背景。
        * 目标陈述：需要评估的句子或陈述，可能由多个句子组成。
        * 任务目标：对每个目标陈述中的句子，找出源文本中与之信息重叠的最匹配句子，并给出一个0到10的分数（0表示无重叠，10表示完全重叠）。如果找不到匹配的句子，请标注为“未找到原句”。

    回答格式：

    请按照以下格式给出你的回答，使用JSON结构：
    [
        {
            "statement": "<目标陈述中的句子>",
            "evidence": "<从源文本中选择可以回答该陈述的原句,如果没有匹配的,请说未找到原句>",
            "score": <输出一个0-10之间的分数,其中0表示没有信息重叠,10表示所有信息都重叠>
        },
        ...
    ]

    示例1:

    源文本: 小明是一个小学生。小明每天早上7点起床,8点到学校上课。放学后,小明先写作业,再看一会儿电视。晚上10点小明准时上床睡觉。
    目标陈述: 小明是一个初中生。他每天6点起床,7点到学校。放学后先玩游戏,再写作业。晚上11点睡觉。    
    回答：
    [
    {
        "statement": "小明是一个初中生。",
        "evidence": "未找到原句",
        "score": 0
    },
    {
        "statement": "他每天6点起床,7点到学校。",
        "evidence": "小明每天早上7点起床,8点到学校上课。",
        "score": 5
    },
    {
        "statement": "放学后先玩游戏,再写作业。",
        "evidence": "放学后,小明先写作业,再看一会儿电视。",
        "score": 3
    },
    {
        "statement": "晚上11点睡觉。",
        "evidence": "晚上10点小明准时上床睡觉。",
        "score": 3
    }
    ]

    示例2:

    源文本: 阿里巴巴集团创立于1999年,是一家以数字化平台为核心的多元创新型企业。公司的使命是让天下没有难做的生意。
    目标陈述: 阿里巴巴成立于1999年,旨在帮助商家更容易做生意。公司总部位于杭州。
    回答：
    [
        {
            "statement": "阿里巴巴成立于1999年,",
            "evidence": "阿里巴巴集团创立于1999年,",
            "score": 10 
        },
        {
            "statement": "旨在帮助商家更容易做生意。",
            "evidence": "公司的使命是让天下没有难做的生意。",
            "score": 9
        },
        {  
            "statement": "公司总部位于杭州。",
            "evidence": "未找到原句",
            "score": 0
        }
    ]
    """
                        
    FAITHFULNESS_USER_PROMPT = """你的实际任务:                       
                            源文本: {contexts}
                            目标陈述: {answer}
                            回答：
                            """
    
    GROUNDEDNESS_SYSTEM_PROMPT = FAITHFULNESS_SYSTEM_PROMPT
    GROUNDEDNESS_USER_PROMPT = """你的实际任务:                       
                            源文本: {answer}
                            目标陈述: {ground_truth}
                            回答：
                            """
    
    CONTEXT_RECALL_SYSTEM_PROMPT = FAITHFULNESS_SYSTEM_PROMPT
    CONTEXT_RECALL_USER_PROMPT = """你的实际任务:                       
                            源文本: {contexts}
                            目标陈述: {ground_truth}
                            回答：
                            """
    # 用户的问题和 RAG 系统最终生成的回答相关度，避免“答非所问”
    ANSWER_RELEVANCE_SYSTEM_PROMPT = """你是一个相关性评分员，为给定的回答对给定的提示进行解释的同时提供相关性评分。
    只能以0到10之间的数字作为回答，其中0表示最不相关，10表示最相关。

    请参照下面的示例以JSON格式进行回答：

    示例1：

    提示：太阳为什么是圆的?
    回答：因为太阳是一个球体,从地球上看来就像一个圆。
    解释和评分：
    [
        {"reason": "回答直接且准确地解释了问题的核心，完全相关。", "score": 10}
    ]

    示例2：

    提示：鸟类有哪些特征?
    回答：鸟类通常有羽毛、喙、翅膀,它们产卵繁殖后代。
    解释和评分：
    [
        {"reason": "回答直接且准确地解释了问题的核心，完全相关。", "score": 10}
    ]

    示例3：

    提示：木星有多少颗卫星?
    回答：木星的卫星有冥王星、土卫六、木卫一等。
    解释和评分：
    [
        {"reason": "回答与问题描述部分相关，但未回答问题的核心。", "score": 4}
    ]

    示例4：

    提示：如何证明黎曼猜想?
    回答：我不知道，因为黎曼猜想是一个尚未解决的数学问题。
    解释和评分：
    [
        {"reason": "回答虽然不完全正确，但是明确表明了不知道。", "score": 10}
    ]

    示例5：

    提示：火星上有生命吗?
    回答：火星大气中含有二氧化碳。
    解释和评分：
    [
        {"reason": "回答只提到了火星的大气成分，与问题无关。", "score": 0}
    ]

    评分指南：
    1. 无论答案长度，只要准确回答问题即可获得高分。
    2. 明确表示不知道或无法回答的情况也视为高度相关，应给予高分。
    3. 答案需与问题的主要内容相关才能获得满分（10分）。
    4. 随着答案与问题相关信息的增加，评分应逐渐提高。
    5. 完全不相关的答案应得0分。
    6. 部分相关的答案根据相关程度可获得2至4分。
    7. 大部分相关的答案根据相关程度可获得5至8分。
    8. 完全相关的答案应获得9分或10分。
    9. 完整且准确回答问题的答案应获得10分。
    10. 明显错误的答案应获得0分。
    11. 表面相关但实际上脱离问题核心的答案也应获得0分。"""

    ANSWER_RELEVANCE_USER_PROMPT = """你的实际任务:                            
                                提示: {question}
                                回答: {answer}
                                解释和评分：
                                """

    # 评估召回文本的精确度——针对检索部分
    CONTEXT_PRECISION_SYSTEM_PROMPT = """作为一个相关性评分员，你的任务是基于给定上下文，对一个特定问题的相关性进行评分。
    你需要根据上下文和问题的关联程度，使用0到10的分数来评价。分数越高，表示上下文与问题的相关性越强。

    请参照下面的示例以JSON格式进行回答：

    示例1：

    上下文：阿尔伯特·爱因斯坦（1879年3月14日 - 1955年4月18日）是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大、最有影响力的科学家之一。
    他最著名的是发展了相对论理论，也对量子力学做出了重要贡献，因此是现代物理学在二十世纪前几十年所取得的革命性重塑自然科学理解的核心人物。他的质能方程E = mc²源自相对论理论，被称为“世界上最著名的方程式”。
    他因“对理论物理学的贡献，特别是对光电效应法的发现”而获得1921年的诺贝尔物理学奖，这是量子理论发展的关键一步。他的工作也因其对科学哲学的影响而闻名。
    在英国《物理世界》杂志1999年的一项针对全球130位领先物理学家的调查中，爱因斯坦被评为有史以来最伟大的物理学家。他的智慧成就及独创性使爱因斯坦成为了天才的代名词。
    问题：你能告诉我阿尔伯特·爱因斯坦的事情吗？
    解释和评分：
    [
        {
            "reason": "所提供的上下文确实有助于得出给定的答案。上下文包括有关阿尔伯特·爱因斯坦的生平和贡献的关键信息，这些信息反映在答案中。",
            "score": 10
        }
    ]


    示例2：

    上下文：安第斯山脉是世界上最长的大陆山脉，位于南美洲。它横跨七个国家，拥有西半球许多最高的山峰。该山脉以其多样的生态系统而闻名，包括高海拔的安第斯高原和亚马逊雨林。
    问题：世界上最高的山峰是哪座？
    解释和评分：
    [
        {
            "reason": "提供的上下文讨论了安第斯山脉，虽然令人印象深刻，但并未包括珠穆朗玛峰或直接关联到关于世界上最高的山峰的问题。",
            "score": 0
        } 
    ]


    评分指南：
    1. 随着上下文为问题提供更多相关背景,相关性得分应该提高。
    2. 随着上下文为问题的更多部分提供相关背景,相关性得分应该提高。
    3. 与问题部分相关的上下文应该得到2、3或4分。得分越高表示相关性越强。
    4. 与问题大部分相关的上下文应该得到5、6、7或8分。得分越高表示相关性越强。
    5. 与整个问题相关的上下文应该得到9或10分。得分越高表示相关性越强。
    6. 上下文必须相关并有助于回答整个问题才能得到10分。
    """

    CONTEXT_PRECISION_USER_PROMPT = """你的实际任务:                               
                                    上下文: {contexts}
                                    问题: {question}
                                    解释和评分：
                                    """
                                    
