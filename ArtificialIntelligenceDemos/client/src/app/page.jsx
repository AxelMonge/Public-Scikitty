import ToolCard from '@/app/components/ToolCard'
import { DEMOS } from './constants/DEMOS';

const Home = () => {
  return (
    <main>
      <article className="px-10 pt-10 text-white mb-5">
        <section className="mx-10 text-white">
          <h1 className="text-7xl text-center text-blue-400">
            Artificial Intelligence Demos
          </h1>
          <p className="mt-8">
            On this website, we demonstrate with examples the functionality of machine learning
            techniques implemented from scratch by students of the elective Artificial Intelligence
            course EIF420O at the Universidad Nacional de Costa Rica. Techniques implemented include
            creating models such as Decision Trees, Tree Gradient Boosting, Rule-Based AI with informed
            search, and Linear and Logistic Regression. The goal is for this website to intuitively showcase
            prediction results in a user-friendly manner.
          </p>
          <div className="flex flex-wrap gap-5 mt-5 justify-around mb-10">
            {DEMOS.map((demo, index) => (
              <ToolCard name={demo.name} image={demo.image} url={demo.url} key={index} />
            ))}
          </div>
        </section>
      </article>
    </main>
  );
}

export default Home;
