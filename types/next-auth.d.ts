import NextAuth from "next-auth"

declare module "next-auth" {
    interface Session {
        user: {
            id: string
            name?: string | null
            email?: string | null
            image?: string | null
        }
    }

    interface User {
        id: string
        name?: string | null
        email?: string | null
        image?: string | null
    }
}

declare module "next-auth/jwt" {
    interface JWT {
        id: string
        /** Epoch ms of the last request that touched this session (idle timeout). */
        lastActivity?: number
        /** Set once the session was dropped for inactivity; forces a re-login. */
        idleExpired?: boolean
    }
}
