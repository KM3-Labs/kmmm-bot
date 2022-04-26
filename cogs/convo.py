from discord.ext import commands
from shimeji import ChatBot
import ray

from utils.bridge import ServalModelProvider


class Convo(commands.Cog):  # maybe separate clm and mlm functions into different cogs?
    def __init__(self, bot):
        self.bot = bot
        self.convutil = ChatBot(
            name="Bib",
            model_provider=ServalModelProvider(  # get settings from settings mnanager like dynaconf... skip the whole model provider thing and decouple from shimeji when convenient, especially since shimeji.ChatBot doesn't have any memory related code.
                address="auto"  # auto if locally launched ray cluster, None if starting from zero (not recommend)
            )
        )
        # TODO: priority channel in db, memories

    @commands.Cog.listener("on_message")
    async def talk(self, message):
        pass

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
