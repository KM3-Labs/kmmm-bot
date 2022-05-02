import logging
import re
import traceback
from urllib import response
import discord
from discord.ext import commands
from shimeji import ChatBot
from shimeji.preprocessor import ContextPreprocessor
from shimeji.postprocessor import NewlinePrunerPostprocessor
from shimeji.util import ContextEntry, INSERTION_TYPE_NEWLINE, TRIM_DIR_TOP, TRIM_TYPE_SENTENCE
import ray

from cogs.utils.bridge import ServalModelProvider
from cogs.utils.chat_util import anti_spam, cut_trailing_sentence
import config

class Convo(commands.Cog):  # maybe separate clm and mlm functions into different cogs?
    def __init__(self, bot):
        self.bot = bot
        self.convutil = ChatBot(
            name="Bib",
            model_provider=ServalModelProvider(  # get settings from settings mnanager like dynaconf... skip the whole model provider thing and decouple from shimeji when convenient, especially since shimeji.ChatBot doesn't have any memory related code.
                address="auto",  # auto if locally launched ray cluster, None if starting from zero (not recommend)
                gpt_model=config.gpt_name,
                bert_model=config.bert_name
            ),
            preprocessors=[ContextPreprocessor(924)],
            postprocessors=[NewlinePrunerPostprocessor()]
        )

        # move below items to a centralised config
        self.context_size = 924
        self.nicknames = ["bib"]
        self.prompt = "[Bib has cat ears.]\n"
        # TODO: priority channel in db
    
    async def get_msg_ctx(self, channel):
        messages = await channel.history(limit=40).flatten()
        messages, to_remove = anti_spam(messages)
        if to_remove:
            logging.info(f'Removed {to_remove} messages from the context.')
        chain = []
        for message in reversed(messages):
            if not message.embeds and message.content:
                content = re.sub(r'\<[^>]*\>', '', message.content)
                if content != '':
                    chain.append(f'{message.author.name}: {content}')
                continue
            elif message.embeds:
                content = message.embeds[0].description
                if content != '':
                    chain.append(f'{message.author.name}: [Embed: {content}]')
                continue
            elif message.attachments:
                chain.append(f'{message.author.name}: [Image attached]')
        return '\n'.join(chain)
    
    async def build_ctx(self, conversation):
        contextmgr = ContextPreprocessor(self.context_size)

        prompt = self.prompt
        prompt_entry = ContextEntry(
            text=prompt,
            prefix='',
            suffix='\n',
            reserved_tokens=512,
            insertion_order=1000,
            insertion_position=-1,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False
        )
        contextmgr.add_entry(prompt_entry)

        # TODO memories
        
        # conversation
        conversation_entry = ContextEntry(
            text=conversation,
            prefix='',
            suffix=f'\n{self.bot}:',
            reserved_tokens=512,
            insertion_order=0,
            insertion_position=-1,
            trim_direction=TRIM_DIR_TOP,
            trim_type=TRIM_TYPE_SENTENCE,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False
        )
        contextmgr.add_entry(conversation_entry)

        return contextmgr.context(self.context_size)

    async def respond(self, conversation, message: discord.Message):
        async with message.channel.typing():
            conversation = await self.build_ctx(conversation)
            response = await self.convutil.respond_async(conversation, push_chain=False)
            response = cut_trailing_sentence(response)
        
        response = response.lstrip()

        await message.channel.send(response)

    @commands.Cog.listener("on_message")
    async def talk(self, message: discord.Message):
        if message.author.id == self.bot.user.id:
            return
        
        message_lower = message.content.lower()
        try:
            if self.bot.user.mentioned_in(message) or any(nick in message_lower for nick in self.nicknames):
                conversation = await self.get_msg_ctx(message.channel)
                await self.respond(conversation, message)
        # TODO conditional response
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())
            embed = discord.Embed(
                title='Error',
                description=str(f'**Exception:** **``{repr(e)}``**\n```{traceback.format_exc()}```'),
            )
            await message.channel.send(embed=embed)

    @commands.Cog.listener("on_ready")
    async def setup_serval(self):
        await self.bot.wait_until_ready()

    def cog_unload(self):
        if ray.is_initialized():
            print("ray is initialized. shutting down with cog...")
            ray.shutdown()
            del self.convutil
        
        return super().cog_unload()


def setup(bot):
    bot.add_cog(Convo(bot))
