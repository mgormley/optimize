package edu.jhu.util;

import java.util.Random;

public class Prng {
    
    public static Random curRandom;
    public static Random javaRandom;
    public static long seed;
    
    public static void seed(long seed) {
        Prng.seed = seed;
        System.out.println("SEED="+seed);
        javaRandom = new Random(seed);
        setRandom(javaRandom);
    }

    public static void setRandom(Random curRandom) {
        Prng.curRandom = curRandom;
    }

    static {
        long DEFAULT_SEED = 123456789101112l;
        //DEFAULT_SEED = System.currentTimeMillis();
        System.out.println("WARNING: pseudo random number generator is not thread safe");
        seed(DEFAULT_SEED);
    }
    
    
    public static double nextDouble() {
        return curRandom.nextDouble();
    }
    
    public static boolean nextBoolean() {
        return curRandom.nextBoolean();
    }
    
    public static int nextInt(int n) {
        return curRandom.nextInt(n);
    }
    
}
