from discord.ext import commands
from cogs.utils import checks


class Test(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name='ping')
    @checks.is_owner()
    async def ping_pong(self, ctx):
        await ctx.send('Pong!')

    @commands.command(name='pong')
    @checks.is_owner()
    async def pong_ping(self, ctx):
        await ctx.send('Ping!')


def setup(bot):
    bot.add_cog(Test(bot))
