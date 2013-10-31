package edu.jhu.hlt.util;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well44497b;

public class Prng {
    
    public static RandomGenerator rng;
    public static long seed;
    
    public static void seed(long seed) {
        Prng.seed = seed;
        System.out.println("SEED="+seed);
        RandomGenerator gen = new Well44497b(seed);
        setRandom(gen);
    }

    public static void setRandom(RandomGenerator rng) {
        Prng.rng = rng;
    }
    
    public static RandomGenerator getRandom() {
    	return Prng.rng;
    }

    static {
        long DEFAULT_SEED = 123456789101112l;
        //DEFAULT_SEED = System.currentTimeMillis();
        System.out.println("WARNING: pseudo random number generator is not thread safe");
        seed(DEFAULT_SEED);
    }
    
    
    public static double nextDouble() {
        return rng.nextDouble();
    }
    
    public static boolean nextBoolean() {
        return rng.nextBoolean();
    }
    
    public static int nextInt(int n) {
        return rng.nextInt(n);
    }
    
}
