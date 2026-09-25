import NextAuth, { type Session } from "next-auth"
import type { NextAuthOptions } from "next-auth"
import CredentialsProvider from "next-auth/providers/credentials"
import { PrismaAdapter } from "@next-auth/prisma-adapter"
import { compare } from "bcryptjs"
import prisma from "@/lib/prisma"

/**
 * Idle session timeout, in minutes.
 *
 * Users stay signed in while they are active: the JWT strategy re-issues the
 * session cookie on every session refresh, so `maxAge` acts as a sliding idle
 * window. Once someone has been away longer than this, the token and cookie
 * expire and the `lastActivity` check below drops the session server-side, so
 * the app sends them back to /login to sign in again.
 *
 * Read from both this route and the client (lib/config.ts), which is why the
 * variable carries the NEXT_PUBLIC_ prefix; it holds no secret.
 */
const SESSION_IDLE_TIMEOUT_MINUTES = Math.max(
    1,
    Number(process.env.NEXT_PUBLIC_SESSION_IDLE_TIMEOUT_MINUTES || 30),
)
const SESSION_IDLE_TIMEOUT_MS = SESSION_IDLE_TIMEOUT_MINUTES * 60 * 1000

// Not exported: Next.js route files may only export HTTP handlers.
const authOptions: NextAuthOptions = {
    adapter: PrismaAdapter(prisma),
    providers: [
        CredentialsProvider({
            name: "credentials",
            credentials: {
                email: { label: "Email", type: "email" },
                password: { label: "Password", type: "password" }
            },
            async authorize(credentials) {
                if (!credentials?.email || !credentials?.password) {
                    throw new Error("Invalid credentials")
                }

                const user = await prisma.user.findUnique({
                    where: {
                        email: credentials.email
                    }
                })

                if (!user || !user.password) {
                    throw new Error("Invalid credentials")
                }

                const isCorrectPassword = await compare(
                    credentials.password,
                    user.password
                )

                if (!isCorrectPassword) {
                    throw new Error("Invalid credentials")
                }

                return {
                    id: user.id,
                    email: user.email,
                    name: user.name,
                }
            }
        })
    ],
    session: {
        strategy: "jwt",
        // Sliding idle window: refreshed on activity, expires once the user
        // has been inactive for longer than this.
        maxAge: SESSION_IDLE_TIMEOUT_MINUTES * 60
    },
    pages: {
        signIn: "/login",
    },
    callbacks: {
        async jwt({ token, user }) {
            const now = Date.now()

            // Fresh sign-in: start the idle clock.
            if (user) {
                token.id = user.id
                token.lastActivity = now
                token.idleExpired = false
                return token
            }

            const lastActivity = typeof token.lastActivity === "number" ? token.lastActivity : now
            if (now - lastActivity > SESSION_IDLE_TIMEOUT_MS) {
                // Away for too long: invalidate the session so the user has to
                // sign in again instead of silently resuming.
                token.idleExpired = true
                return token
            }

            token.lastActivity = now
            token.idleExpired = false
            return token
        },
        async session({ session, token }) {
            if (token.idleExpired) {
                // An empty session makes next-auth report "unauthenticated" on
                // the client, which sends the user to /login.
                return {} as Session
            }
            if (session.user) {
                session.user.id = token.id as string
            }
            return session
        }
    },
    secret: process.env.NEXTAUTH_SECRET,
}

const handler = NextAuth(authOptions)

export { handler as GET, handler as POST }
